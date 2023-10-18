
class DebugSeq2SeqTrainer(Seq2SeqTrainer):
    def non_sense(self):
        def evaluate(
                self,
                eval_dataset: Optional[Dataset] = None,
                ignore_keys: Optional[List[str]] = None,
                metric_key_prefix: str = "eval",
                **gen_kwargs,
        ) -> Dict[str, float]:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)

            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
            for data in eval_dataloader:
                with open('debug.txt', 'w', encoding='utf-8') as O:
                    print(data)
                gg

        def get_train_dataloader(self) -> DataLoader:
            dataloader = super().get_train_dataloader()
            for data in dataloader:
                print(data)
                gg

        def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
            print(inputs)
            gg

        def compute_loss(self, model, inputs, return_outputs=False):
            print(f'{"-" * 40} debug {"-" * 40}')
            for key in inputs:
                print(key)
            return super().compute_loss(model, inputs, return_outputs)


class DebugDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        # with open('./debug.txt', 'a', encoding='utf-8') as O:
        #     print(features[0], file=O)
        # with open('./debug_all.txt', 'w', encoding='utf-8') as O:
        #     print(features, file=O)
        labels = [feature["labels"] for feature in features] \
            if hasattr(features[0], 'keys') and "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features