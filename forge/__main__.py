"""CLI for model preparation: python -m forge.model_prep or vllm-factory-prep."""

import argparse
import sys

from forge.model_prep import prepare_gliner2_model, prepare_gliner25_model


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vllm-gliner-prep",
        description="Prepare GLiNER-family models for vLLM serving",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g. knowledgator/gliner-bi-large-v2.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: ./<model-slug>-vllm)",
    )
    parser.add_argument(
        "--plugin",
        default=None,
        choices=["deberta_gliner2", "deberta_gliner25"],
        help="Explicit plugin type (auto-detected from model metadata if omitted)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
    )
    args = parser.parse_args()

    try:
        if not args.plugin:
            parser.error("--plugin is required (deberta_gliner2 or deberta_gliner25)")
        prepare = (
            prepare_gliner25_model if args.plugin == "deberta_gliner25" else prepare_gliner2_model
        )
        prepare(hf_model_id=args.model, output_dir=args.output, force=args.force)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
