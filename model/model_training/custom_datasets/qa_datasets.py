class AlpacaGpt4(Dataset):
    def __init__(self, cache_dir: str | Path, mode: str = "sft", input_max_length: int = 2048) -> None:
        super().__init__()
        self.rows = []
        if mode not in ("sft", "rl"):
            raise NotImplementedError(f"Currently only the modes 'sft' and 'rl' are implemented. Received {mode}.")
        self.mode = mode
        data = load_dataset("vicgalle/alpaca-gpt4", cache_dir=cache_dir)
        for line in data["train"]:
            if (conv := self._process_instruction(line, input_max_length)) is not None:
                self.rows.append(conv)

    def _process_instruction(self, row: dict[str, str], input_max_length: int) -> list[str] | None:
        # discard items that are too long: when checked on 2023-04-17 this was just one item in the whole dataset with length above 2048.
        # And 12 above 1024.
        if len(row["input"]) + len(row["instruction"]) > input_max_length:
            return None
        # filter all appearing variants of "no input" or empty input or cases where the input is already in the instruction.
        # In this cases we don't add the input
        if (
            any([k in row["input"].lower() for k in ["no input", "noinput", "n/a"]])
            or (not row["input"])
            or (row["input"].lower() in row["instruction"].lower())
        ):
            return [row["instruction"], row["output"]]
        # Concatenate the instruction and input.
        else:
            return [f"{row['instruction']} {row['input']}", row["output"]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> list[str] | tuple[str]:
        dialogue: list[str] = self.rows[index]
        if self.mode == "sft":
            return dialogue
        elif self.mode == "rl":
            return tuple(dialogue[:-1])