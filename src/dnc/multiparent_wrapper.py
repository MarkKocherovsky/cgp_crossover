from dnc.multiparent_dnc import NeuralCrossover
import torch
import traceback
import numpy as np
from copy import deepcopy
class NeuralCrossoverWrapper:
    def __init__(self, embedding_dim, sequence_length, input_dim, get_fitness_function, running_mean_decay=0.99,
                 batch_size=32, load_weights_path=None, freeze_weights=False, learning_rate=1e-3, epsilon_greedy=0.1,
                 use_scheduler=False, use_device='cpu', adam_decay=0, clip_grads=False, n_parents=2, crossover_type='uniform', n_points=1):
        self.device = use_device
        self.neural_crossover = NeuralCrossover(embedding_dim, embedding_dim, input_dim, sequence_length,
                                                n_parents=n_parents, device=use_device).to(
            self.device)
        self.running_mean_decay = running_mean_decay
        self.optimizer = torch.optim.Adam(self.neural_crossover.parameters(), lr=learning_rate, weight_decay=adam_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, verbose=True)
        self.get_fitness_function = get_fitness_function
        self.batch_size = batch_size
        self.n_parents = n_parents
        self.batch_stack_fitness_values = []
        self.sampled_action_space = []
        self.sampled_solutions = []
        self.load_weights_path = load_weights_path
        self.freeze_weights = freeze_weights
        self.epsilon_greedy = epsilon_greedy
        self.use_scheduler = use_scheduler
        self.clip_grads = clip_grads
        self.acc_batch_length = 0
        self.crossover_mode = crossover_type
        self.n_points = n_points

        if self.load_weights_path is not None:
            self.neural_crossover.load_state_dict(torch.load(self.load_weights_path))

    def get_batch_and_clear(self, verbose:bool = False):
        """
        Returns the batch of parents and fitness values and clears the batch.
        """
        if verbose:
            for i, t in enumerate(self.sampled_action_space):
                print(f"🧪 sampled_action_space[{i}] shape: {t.shape}")
            for i, t in enumerate(self.sampled_solutions):
                print(f"🔍 sampled_solutions[{i}] shape: {t.shape}")


        fitness_values = torch.cat(self.batch_stack_fitness_values, dim=0).unsqueeze(1).to(self.device)
        sampled_action_space = torch.cat(self.sampled_action_space, dim=0).to(self.device)
        sampled_solutions = torch.cat(self.sampled_solutions, dim=0).to(self.device)
        self.clear_stacks()

        return fitness_values, sampled_action_space, sampled_solutions

    def clear_stacks(self):
        """
        Clears the batch stacks.
        """
        self.batch_stack_fitness_values.clear()
        self.sampled_action_space.clear()
        self.sampled_solutions.clear()

    def run_epoch(self):
        """
        Performs one step of training on the neural crossover.
        """
        if self.freeze_weights:
            self.clear_stacks()
            return

        total_batches_length = self.acc_batch_length
        if total_batches_length < self.batch_size:
            return

        self.acc_batch_length = 0

        fitness_values, sampled_action_space, sampled_solutions = self.get_batch_and_clear()
        self.optimizer.zero_grad()
        sampled_solutions_proba = torch.gather(sampled_action_space, 1, sampled_solutions.unsqueeze(1)).squeeze(1)
        # Assume you have B models and S sequences, then:
        S = sampled_solutions_proba.shape[0] // fitness_values.shape[0]
        fitness_values = fitness_values.repeat_interleave(S, dim=0)

        if torch.any(sampled_solutions_proba <= 0):
            print("⚠️ Warning: sampled_solutions_proba has non-positive entries:", sampled_solutions_proba)
            sampled_solutions_proba = sampled_solutions_proba.clamp(min=1e-8)
        #print(f"🧪 sampled_solutions_proba shape: {sampled_solutions_proba.shape}")
        #print(f"🧪 fitness_values shape: {fitness_values.shape}")
        assert sampled_solutions_proba.shape[0] == fitness_values.shape[0], \
            f"Mismatch: got {sampled_solutions_proba.shape[0]} samples but {fitness_values.shape[0]} fitness values"

        loss = -torch.mean(
            torch.log(sampled_solutions_proba) * (fitness_values.type(torch.DoubleTensor)).to(self.device))

        loss.backward()

        if self.clip_grads:
            torch.nn.utils.clip_grad_norm_(self.neural_crossover.parameters(), 1.0)

        self.optimizer.step()

        if self.use_scheduler:
            self.scheduler.step(loss)

        #print(f'loss: {loss}, reward: {torch.mean(fitness_values.type(torch.DoubleTensor))}')


    def combine_parents_uniform(self, parents_matrix):
        if self.freeze_weights:
            self.neural_crossover.eval()

        parents_matrix = parents_matrix.to(self.device)
        attention_values, selected_crossovers_indices = self.neural_crossover(
            parents_matrix,
            epsilon_greedy=self.epsilon_greedy
        )
        #print(f'attention_values shape before append: {attention_values.shape}')

        # attention_values shape: [batch, seq_len, n_parents]
        # selected_crossovers_indices shape: [batch, seq_len]
        B, S, P = attention_values.shape
        assert attention_values.ndim == 3
        assert P == 2
        assert selected_crossovers_indices.shape == (B, S)

        # ✅ Flatten to 2D for training
        attention_values = attention_values.view(B * S, P)
        selected_crossovers_indices = selected_crossovers_indices.view(B * S)

        # 🔒 Append only after reshape
        self.sampled_action_space.append(attention_values)
        self.sampled_solutions.append(selected_crossovers_indices)

        #print(f"✅ appended attention_values shape: {attention_values.shape}")

        # Continue with crossover...
        s_i = np.row_stack(np.array(deepcopy(selected_crossovers_indices)).astype(int))
        pm = parents_matrix.permute(1, 2, 0, 3)  # (batch, seq_len, parents, gene_dim)
        idx = selected_crossovers_indices.view(B, S).unsqueeze(-1).unsqueeze(-1)
        gene_dim = parents_matrix.shape[-1]
        idx = idx.expand(-1, -1, 1, gene_dim)
        gathered = torch.gather(pm, dim=2, index=idx).squeeze(2)
        return gathered, s_i, attention_values, selected_crossovers_indices


    def combine_parents_n_point(self, parents_matrix):
        """
        Performs learned n-point crossover using the DNC to generate crossover points.
        """
        if self.freeze_weights:
            self.neural_crossover.eval()

        parents_matrix = parents_matrix.to(self.device)
        batch_size, seq_len = parents_matrix.shape[1], parents_matrix.shape[2]
        n_parents = parents_matrix.shape[0]

        attention_values, selected_crossovers_indices = self.neural_crossover(parents_matrix, epsilon_greedy=self.epsilon_greedy)

        # For training:
        self.sampled_action_space.append(attention_values)
        self.sampled_solutions.append(selected_crossovers_indices)

        # Result container
        children = []

        # Process each child in the batch
        for b in range(batch_size):
            # Sampled positions for this child
            crossover_points = selected_crossovers_indices[b].detach().cpu().numpy()

            # Select only the first `n_points` unique, sorted points
            unique_points = np.unique(crossover_points)
            cut_points = np.sort(unique_points[:self.n_points])

            # Ensure boundary at start and end
            boundaries = np.concatenate(([0], cut_points, [seq_len]))

            # Alternate parent segments
            new_child = []
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                parent_id = i % n_parents
                segment = parents_matrix[parent_id, b, start:end, :]  # ✅ RIGHT — keeps each gene's 6 features
                new_child.append(segment)

            # Concatenate segments into full child
            new_child = torch.cat(new_child, dim=0)
            children.append(new_child)

        # Stack into tensor: shape (batch_size, seq_len)
        result = torch.stack(children, dim=0)

        # DNC returns sampled points for logging/debugging
        crossover_indices_np = np.row_stack(np.array(deepcopy(selected_crossovers_indices)).astype(int))
        return gathered, s_i, attention_values, selected_crossovers_indices


    def update_batch_stack(self, fitness_values):
        """
        Updates the batch stack.
        """
        fitness_values = fitness_values.view(-1)  # Ensures it's 1D
        self.batch_stack_fitness_values.append(fitness_values)

    def get_crossover(self, parents_matrix, x, y):
        """
        Uses the neural crossover to select the crossover points from the parents.
        Then performs one step of training on the neural crossover.
        :param parents_matrix: parents to crossover
        :return: resulting crossover individuals
        """
        parents_matrix = torch.Tensor(parents_matrix).type(torch.LongTensor)
        if self.crossover_mode == 'uniform':
            selected_crossover_func = self.combine_parents_uniform
        elif self.crossover_mode == 'n_point':
            selected_crossover_func = self.combine_parents_n_point
        else:
            raise ValueError(f"Unknown crossover_mode: {self.crossover_mode}")
        
        # Generate child1 (used for training)
        child1, distro1, attention_values1, selected_crossovers_indices1 = selected_crossover_func(parents_matrix)

        # Generate child2 (diversity only)
        child2, distro2 = selected_crossover_func(parents_matrix)[:2]  # discard extra outputs if any

        # Only log child1 data
        self.sampled_action_space.append(attention_values1)
        self.sampled_solutions.append(selected_crossovers_indices1)

        child1_fitness_values = [self.get_fitness_function(x, y) for child in child1.detach().cpu().numpy()]
        child1_fitness_values = torch.Tensor(child1_fitness_values).type(torch.FloatTensor)
        self.update_batch_stack(child1_fitness_values)

        
        self.run_epoch()
 
        return child1.detach().cpu().numpy(), child2.detach().cpu().numpy(), distro1, distro2

    def cross_pairs(self, parents_pairs, x, y):
        if len(parents_pairs) == 0:
            return []
        # Convert each parent to tensor, then group by parent index
        n_parents = len(parents_pairs[0])
        batch_size = len(parents_pairs)

        # Shape: (n_parents, batch_size, seq_len, gene_dim)
        parents_matrix = torch.stack([
            torch.stack([
                torch.tensor(parents_pairs[j][i].model, dtype=torch.float32)
                for j in range(batch_size)
            ], dim=0)
            for i in range(n_parents)
        ], dim=0)
        
        #parents_matrix = parents_matrix.unsqueeze(0)
        self.acc_batch_length += parents_matrix.shape[1]
        child1, child2, distro1, distro2 = self.get_crossover(parents_matrix, x, y)
        child1 = np.array(child1)
        child2 = np.array(child2)
        return list(zip(child1, child2)), np.concatenate((distro1, distro2), axis = 0)

    def save_weights(self, path):
        torch.save(self.neural_crossover.state_dict(), path)
