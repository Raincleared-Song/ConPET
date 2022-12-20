class ChoiceEvaluator:
    def __init__(self, true_label_num: int = -1, choice_num: int = -1, per_sample: int = -1):
        self.true_label_num = true_label_num
        self.choice_num = choice_num
        self.per_sample = per_sample if per_sample > 0 else (true_label_num + choice_num - 1) // choice_num

    def get_choice_accuracy(self, preds: list, labels: list):
        assert self.per_sample > 0
        assert len(preds) == len(labels) and len(preds) % self.per_sample == 0
        correct_num = 0
        true_sample_num = len(preds) // self.per_sample
        for idx in range(true_sample_num):
            cur_preds = preds[idx * self.per_sample:(idx + 1) * self.per_sample]
            cur_labels = labels[idx * self.per_sample:(idx + 1) * self.per_sample]
            correct_num += int(cur_preds == cur_labels)
        return round(correct_num / true_sample_num * 100, 2), correct_num, true_sample_num
