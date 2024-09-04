import torch

def concat_text_input_output(input_ids, input_atts, output_ids, output_atts):
    input_part_targets_len = []
    llm_tokens = {"input_ids": [], "attention_mask": []}
    for i in range(input_ids.size(0)):
        this_input_ones = input_atts[i].sum()
        input_part_targets_len.append(this_input_ones)
        llm_tokens["input_ids"].append(
            torch.cat(
                [
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:],
                ]
            )
        )
        llm_tokens["attention_mask"].append(
            torch.cat(
                [
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:],
                ]
            )
        )
    llm_tokens["input_ids"] = torch.stack(llm_tokens["input_ids"])
    llm_tokens["attention_mask"] = torch.stack(llm_tokens["attention_mask"])
    return llm_tokens, input_part_targets_len