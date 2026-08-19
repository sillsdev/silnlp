"""Interactively chat with a fine-tuned LLM checkpoint from the terminal -- useful for spot
checking how well general instruction-following survived translation fine-tuning, since that
isn't something the translation eval/test metrics can tell you.
"""

import argparse

import torch

from silnlp.common.environment import SilNlpEnv
from silnlp.nmt.config_utils import load_config
from silnlp.nmt.llm_config import TRANSLATE_GEMMA_MODEL_PREFIXES, LLMConfig, LLMModel

EXIT_COMMANDS = {"/exit", "/quit"}
RESET_COMMANDS = {"/reset", "/new"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned LLM checkpoint")
    parser.add_argument("experiment", help="Experiment name")
    parser.add_argument("--checkpoint", default="last", help="Checkpoint to use (last, best, or a checkpoint step #)")
    parser.add_argument("--system-message", default="", help="Optional system message for the conversation")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate per reply")
    parser.add_argument(
        "--temperature", type=float, default=None, help="Sampling temperature; omit for greedy decoding"
    )
    args = parser.parse_args()

    environment = SilNlpEnv.create_standard_environment()
    config = load_config(args.experiment, environment)
    if not isinstance(config, LLMConfig):
        parser.error(f"Experiment '{args.experiment}' is not an LLM experiment.")

    if config.model.lower().startswith(TRANSLATE_GEMMA_MODEL_PREFIXES):
        print(
            "Warning: this model's chat template only supports its structured translation format "
            "and may not respond sensibly to a plain-text conversation.\n"
        )

    model_wrapper = config.create_model()
    assert isinstance(model_wrapper, LLMModel)
    model, tokenizer = model_wrapper.load_for_inference(args.checkpoint)
    if tokenizer.chat_template is None:
        parser.error(f"The tokenizer for '{config.model}' has no chat template; can't hold a conversation with it.")
    tokenizer.padding_side = "left"

    base_messages = [{"role": "system", "content": args.system_message}] if args.system_message else []
    messages = list(base_messages)

    print(f"Chatting with '{args.experiment}' (checkpoint={args.checkpoint}).")
    print("Commands: /reset to clear history, /exit to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input in EXIT_COMMANDS:
            break
        if user_input in RESET_COMMANDS:
            messages = list(base_messages)
            print("(conversation reset)\n")
            continue

        messages.append({"role": "user", "content": user_input})
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=False
        ).to(model.device)

        gen_kwargs = {"max_new_tokens": args.max_new_tokens, "pad_token_id": tokenizer.pad_token_id}
        if args.temperature is not None:
            gen_kwargs.update(do_sample=True, temperature=args.temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            output = model.generate(input_ids, **gen_kwargs)
        response = tokenizer.decode(output[0][input_ids.shape[-1] :], skip_special_tokens=True).strip()

        print(f"\nAssistant: {response}\n")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
