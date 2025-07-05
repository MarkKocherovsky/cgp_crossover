from dnc.multiparent_dnc import NeuralCrossover
import torch
import traceback
import numpy as np
from copy import deepcopy
class NeuralCrossoverWrapper:
    def __init__(self, embedding_dim, sequence_length, input_dim, get_fitness_function, running_mean_decay=0.99,
                 batch_size=820, load_weights_path=None, freeze_weights=False, learning_rate=1e-3, epsilon_greedy=0.1,
                 use_scheduler=False, use_device='cpu', adam_decay=0.95, clip_grads=False, n_parents=2, crossover_type='uniform', n_points=1):
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
    def clear_stacks(self):
        """
        Clears the batch stacks.
        """
        self.batch_stack_fitness_values.clear()
        self.sampled_action_space.clear()
        self.sampled_solutions.clear()

    def get_batch_and_clear(self, verbose: bool = False):
        if verbose:
            for i, t in enumerate(self.sampled_action_space):
                print(f"🧪 sampled_action_space[{i}] shape: {t.shape}")
            for i, t in enumerate(self.sampled_solutions):
                print(f"🔍 sampled_solutions[{i}] shape: {t.shape}")

        fitness_values = torch.cat(self.batch_stack_fitness_values, dim=0).unsqueeze(1).to(self.device)
        print(f'len(self.sampled_action_space): {len(self.sampled_action_space)}')
        print(f'fitness_values.shape: {fitness_values.shape}')

        normalized_action_space = []
        for i, t in enumerate(self.sampled_action_space):
            if t.dim() == 3:
                if verbose:
                    print(f"⚠️ sampled_action_space[{i}] has 3D shape {t.shape}, reshaping")
                t = t.view(-1, t.shape[-1])
            elif t.dim() == 1:
                if verbose:
                    print(f"⚠️ sampled_action_space[{i}] has 1D shape {t.shape}, unsqueezing")
                t = t.unsqueeze(0)
            normalized_action_space.append(t)
        sampled_action_space = torch.cat(normalized_action_space, dim=0).to(self.device)

        normalized_solutions = []
        for i, t in enumerate(self.sampled_solutions):
            if t.dim() > 1:
                if verbose:
                    print(f"⚠️ sampled_solutions[{i}] has shape {t.shape}, flattening")
                t = t.view(-1)
            normalized_solutions.append(t)
        sampled_solutions = torch.cat(normalized_solutions, dim=0).to(self.device)

        self.clear_stacks()
        return fitness_values, sampled_action_space, sampled_solutions

    def run_epoch(self):
        if self.freeze_weights:
            self.clear_stacks()
            return

        if self.acc_batch_length < self.batch_size:
            return

        self.acc_batch_length = 0
        fitness_values, sampled_action_space, sampled_solutions = self.get_batch_and_clear()

        if fitness_values.dim() == 2:
            fitness_values = fitness_values.squeeze(1)

        print(f"sampled_action_space shape: {sampled_action_space.shape}")
        print(f"sampled_solutions shape: {sampled_solutions.shape}")
        print(f"fitness_values shape: {fitness_values.shape}")
        print(f"crossover_type: {self.crossover_mode}, n_points: {self.n_points}")

        B = fitness_values.shape[0]

        if self.crossover_mode == 'uniform':
            # Infer S_fn from shapes: sampled_action_space is [B * S_fn]
            S_fn = sampled_action_space.shape[0] // B
            print(f"🔍 sampled_action_space.shape: {sampled_action_space.shape}")
            print(f"🔍 fitness_values.shape: {fitness_values.shape}")
            print(f"→ B: {B}")
            print(f"→ Implied S_fn: {sampled_action_space.shape[0] // B} (expecting int)")
            print(f"→ sampled_action_space.shape[0] % B = {sampled_action_space.shape[0] % B}")

            assert S_fn * B == sampled_action_space.shape[0], "sampled_action_space not divisible by batch size"

            fitness_values = fitness_values.repeat_interleave(S_fn)  # [B * S_fn]

            if sampled_action_space.shape[0] != fitness_values.shape[0]:
                raise ValueError(
                    f"Mismatched sizes after repeat: {sampled_action_space.shape[0]} vs {fitness_values.shape[0]}")

            sampled_solutions_proba = torch.gather(sampled_action_space, 0, sampled_solutions)

        else:
            n_points = sampled_action_space.shape[0] // B
            fitness_values = fitness_values.repeat_interleave(n_points)
            sampled_solutions_proba = torch.gather(sampled_action_space, 1, sampled_solutions.unsqueeze(1)).squeeze(1)

        if torch.any(sampled_solutions_proba <= 0):
            print("⚠️ Warning: sampled_solutions_proba has non-positive entries. Clamping.")
            sampled_solutions_proba = sampled_solutions_proba.clamp(min=1e-8)

        if sampled_solutions_proba.shape[0] != fitness_values.shape[0]:
            raise ValueError(
                f"Mismatch after gathering: got {sampled_solutions_proba.shape[0]} samples but {fitness_values.shape[0]} fitness values"
            )

        self.optimizer.zero_grad()
        loss = -torch.mean(
            torch.log(sampled_solutions_proba) * fitness_values.to(self.device, dtype=torch.double)
        )
        loss.backward()

        if self.clip_grads:
            torch.nn.utils.clip_grad_norm_(self.neural_crossover.parameters(), 1.0)

        self.optimizer.step()

        if self.use_scheduler:
            self.scheduler.step(loss)

    def get_attention_and_indices(self, parents_matrix, semantic_matrix=None):
        parents_matrix = parents_matrix.to(self.device)
        if semantic_matrix is not None:
            semantic_matrix = semantic_matrix.to(self.device)

        use_uniform = "uniform" in self.crossover_mode.lower()
        use_n_point = "n_point" in self.crossover_mode.lower()

        if use_uniform:
            input_tensor = semantic_matrix if semantic_matrix is not None else parents_matrix
            attention_logits, selected_crossovers_indices = self.neural_crossover(
                input_tensor,
                epsilon_greedy=self.epsilon_greedy
            )  # attention_logits: [B, S_fn, P], selected: [B, S_fn]

            # Collapse attention to parent 1 (P=1), so shape is [B, S_fn]
            attention_values = attention_logits[..., 1]

            return attention_values, selected_crossovers_indices
        
        elif use_n_point:
            input_tensor = semantic_matrix if semantic_matrix is not None else parents_matrix

            attention_logits, _ = self.neural_crossover(input_tensor,
                                                        epsilon_greedy=self.epsilon_greedy)
            B, S_fn, P = attention_logits.shape
            assert P == 2, "Expected 2 parents for n-point crossover"

            attention_weights = torch.softmax(attention_logits.mean(dim=-1), dim=1)
            selected_crossovers_indices = []
            attention_values = torch.zeros(B, self.n_points, S_fn, device=self.device)
            for b in range(B):
                weights = attention_weights[b].clone()
                cut_points = []
                for _ in range(self.n_points):
                    if torch.sum(weights) <= 0:
                        break
                    idx = torch.multinomial(weights, 1).item()
                    cut_points.append(idx)
                    weights[idx] = 0
                cut_points = sorted(cut_points)
                selected_crossovers_indices.append(torch.tensor(cut_points, device=self.device))
                for i, cp in enumerate(cut_points):
                    attention_values[b, i, cp] = 1.0
            selected_crossovers_indices = torch.stack(selected_crossovers_indices)
            return attention_values, selected_crossovers_indices

        else:
            raise ValueError(f"Unsupported crossover type: {self.crossover_mode}")


    @staticmethod
    def extract_function_nodes(matrix):
        """
        Extracts function and output nodes from a batch of individuals.

        Args:
            matrix (torch.Tensor): Tensor of shape (P, B, N, D), where
                P = number of parent sets,
                B = number of parents per set,
                N = number of nodes per individual,
                D = node representation length (e.g., 6)

        Returns:
            torch.Tensor: Tensor of shape (P, B, S_fn, D) where S_fn is the number of
                          function and output nodes per individual.
        """
        mask = (matrix[..., 0] == 2) | (matrix[..., 0] == 3)  # shape: (P, B, N)
        selected = matrix[mask]  # shape: (P * B * S_fn, D)
        expected_total = mask.sum(dim=-1)  # shape: (P, B)

        # Sanity check: All individuals must have same S_fn
        unique_counts = torch.unique(expected_total)
        assert len(unique_counts) == 1, f"Inconsistent S_fn across individuals: {unique_counts.tolist()}"

        S_fn = unique_counts.item()
        return selected.view(matrix.shape[0], matrix.shape[1], S_fn, matrix.shape[-1])  # shape: (P, B, S_fn, D)

    def combine_parents_uniform(self, parents_matrix, semantic_matrix=None):
        if self.freeze_weights:
            self.neural_crossover.eval()

        fn_only = self.extract_function_nodes(parents_matrix)  # [2, B, S_fn, G]
        attention_values, selected_crossovers_indices = self.get_attention_and_indices(fn_only, semantic_matrix)

        assert attention_values.ndim == 2, f"Expected [B, S_fn], got {attention_values.shape}"
        assert selected_crossovers_indices.ndim == 2, f"Expected [B, S_fn], got {selected_crossovers_indices.shape}"

        B, S_fn = attention_values.shape
        assert B == parents_matrix.shape[1], "Batch size mismatch"

        # Flatten for training (match policy gradient expectations)
        print(f"✅ attention_values shape before append: {attention_values.shape}")
        print(f"✅ selected_crossovers_indices shape: {selected_crossovers_indices.shape}")

        self.sampled_action_space.extend(attention_values.detach().reshape(B, -1))
        self.sampled_solutions.append(selected_crossovers_indices.detach().view(-1))  # [B * S_fn]
        print(f"✅ FINAL sampled_action_space length: {len(self.sampled_action_space)}")

        # Prepare crossover output
        _, B_check, S_full, G = parents_matrix.shape
        assert B == B_check
        selected_crossovers_indices = selected_crossovers_indices.view(B, S_fn)

        children = parents_matrix[0].clone()  # [B, S_full, G]

        for b in range(B):
            node_types = parents_matrix[0, b, :, 0]  # [S_full]
            function_indices = ((node_types == 2) | (node_types == 3)).nonzero(as_tuple=True)[0]
            assert len(function_indices) == S_fn, f"Expected {S_fn} function nodes but found {len(function_indices)}"

            for i, node_idx in enumerate(function_indices):
                if selected_crossovers_indices[b, i].item() == 1:
                    children[b, node_idx] = parents_matrix[1, b, node_idx]

        return children, selected_crossovers_indices.cpu().numpy(), attention_values, selected_crossovers_indices

    def combine_parents_n_point(self, parents_matrix, semantic_matrix=None):
        """
        Perform learned n-point crossover using NeuralCrossover module.
        Only applied to function nodes (node_type == 2 or 3).
        Returns:
            children: Tensor [B, S, G]
            s_i: np.ndarray [B, n_points]
            attention_values: Tensor [B * n_points, S_fn]
            selected_crossovers_indices: Tensor [B * n_points]
        """
        if self.freeze_weights:
            self.neural_crossover.eval()

        # Extract function nodes only
        fn_only = self.extract_function_nodes(parents_matrix)  # [2, B, S_fn, G]
        attention_values, selected_crossovers_indices = self.get_attention_and_indices(fn_only, semantic_matrix)

        B, _, n_points = attention_values.shape
        S_fn = len(((parents_matrix[0, 0, :, 0] == 2) | (parents_matrix[0, 0, :, 0] == 3)).nonzero())
        _, _, S_full, G = parents_matrix.shape

        attention_flat = attention_values.view(B * n_points)
        selected_flat = selected_crossovers_indices.view(B)

        self.sampled_action_space.append(attention_flat.detach())
        self.sampled_solutions.append(selected_flat.detach())
        # parents_matrix: [2, B, S_full, G]
        _, B_check, S_full, G = parents_matrix.shape
        assert B == B_check
        children = []

        for b in range(B):
            # Prepare base child from parent 0
            child = parents_matrix[0, b].clone()  # [S_full, G]

            # Get positions of function nodes
            node_types = parents_matrix[0, b, :, 0]
            fn_indices = ((node_types == 2) | (node_types == 3)).nonzero(as_tuple=True)[0]  # [S_fn]

            assert len(fn_indices) == S_fn, f"Mismatch: expected {S_fn} function nodes, found {len(fn_indices)}"

            cut_points = selected_crossovers_indices[b].detach().cpu().numpy()
            cut_points = np.unique(cut_points)
            cut_points = np.sort(cut_points[:self.n_points])
            boundaries = np.concatenate(([0], cut_points, [S_fn]))

            # Build function node segment from alternating parents
            segments = []
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                parent_id = i % 2  # toggle between 0 and 1
                if end > start:
                    fn_slice = fn_indices[start:end]  # actual indices into full genome
                    segment = parents_matrix[parent_id, b, fn_slice, :]  # [len, G]
                    segments.append((fn_slice, segment))

            # Insert segments into child
            for indices, segment in segments:
                child[indices] = segment

            children.append(child)

        children = torch.stack(children, dim=0)  # [B, S_full, G]
        s_i = np.row_stack(np.array(deepcopy(selected_crossovers_indices)).astype(int))  # for logging

        return children, s_i, attention_flat, selected_flat

    def update_batch_stack(self, fitness_values):
        """
        Updates the batch stack.
        """
        fitness_values = fitness_values.view(-1)  # Ensures it's 1D
        self.batch_stack_fitness_values.append(fitness_values)

    def get_crossover(self, parents_matrix, x, y, semantic_matrix):
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
        child1, distro1, attention_values1, selected_crossovers_indices1 = selected_crossover_func(parents_matrix, semantic_matrix)

        # Generate child2 (diversity only)
        child2, distro2 = selected_crossover_func(parents_matrix, semantic_matrix)[:2]  # discard extra outputs if any

        child1_fitness_values = [self.get_fitness_function(x, y) for child in child1.detach().cpu().numpy()]
        child1_fitness_values = torch.Tensor(child1_fitness_values).type(torch.FloatTensor)
        self.update_batch_stack(child1_fitness_values)

        
        self.run_epoch()
 
        return child1.detach().cpu().numpy(), child2.detach().cpu().numpy(), distro1, distro2

    def cross_pairs(self, parents_pairs, x, y, semantic_pairs=None):
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

        semantic_matrix = torch.stack([
            torch.stack([
                torch.tensor(semantic_pairs[j][i], dtype=torch.float32)
                for j in range(batch_size)
            ], dim=0)
            for i in range(n_parents)
        ], dim=0) if semantic_pairs is not None else None

        
        #parents_matrix = parents_matrix.unsqueeze(0)
        self.acc_batch_length += parents_matrix.shape[1]
        child1, child2, distro1, distro2 = self.get_crossover(parents_matrix, x, y, semantic_matrix)
        child1 = np.array(child1)
        child2 = np.array(child2)
        return list(zip(child1, child2)), np.concatenate((distro1, distro2), axis = 0)

    def save_weights(self, path):
        torch.save(self.neural_crossover.state_dict(), path)
