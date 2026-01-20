import torch
import re

class MDDReasoningRewardManager:
    def __init__(self, acc_weight=1.0, fmt_weight=0.5, cons_weight=1.0):
        self.acc_weight = acc_weight
        self.fmt_weight = fmt_weight
        self.cons_weight = cons_weight

    def __call__(self, data):
        """
        veRL passes a data object containing:
        - data.batch.resps: Generated response tokens (need decoding)
        - data.non_tensor_batch: Original metadata (e.g., ground-truth labels)
        """
        # 1. Extract responses and labels
        # Note: Depending on your veRL setup, responses might be pre-decoded in non_tensor_batch
        responses = [item['responses'] for item in data.non_tensor_batch]
        labels = [item['label'] for item in data.non_tensor_batch]
        
        rewards = []
        for response, label in zip(responses, labels):
            # Calculate individual components
            r_acc = self._accuracy_reward(response, label)
            r_fmt = self._format_reward(response)
            r_cons = self._consistency_reward(response)
            
            # Weighted total
            total = (self.acc_weight * r_acc + 
                     self.fmt_weight * r_fmt + 
                     self.cons_weight * r_cons)
            rewards.append(total)
            
        return torch.tensor(rewards, dtype=torch.float32)

    def _accuracy_reward(self, response, label):
        # Use regex to extract diagnosis to prevent the model from gaming the system
        pattern = r"Diagnosis\s*:\s*(.*)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            pred = match.group(1).lower()
            return 1.0 if label.lower() in pred else 0.0
        return 0.0

    def _format_reward(self, response):
        # Ensure the model follows the <Reasoning> and <Diagnosis> structure
        has_reasoning = "Reasoning" in response
        has_diagnosis = "Diagnosis" in response
        return 1.0 if (has_reasoning and has_diagnosis) else 0.0

    def _consistency_reward(self, response):
        # Check for logical contradictions
        resp_lower = response.lower()
        if "no depressive symptoms" in resp_lower and "mdd" in resp_lower:
            return -1.0
        return 0.0