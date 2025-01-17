import argparse
from scheduler import Scheduler
from prompts import prompts
import time
import textwrap


def main():
    parser = argparse.ArgumentParser(description="A simple LLM inference server.")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=4,
        help="The batch size for processing prompts (default: 8)",
    )
    parser.add_argument(
        "-l",
        "--generation-length",
        type=int,
        default=100,
        help="The length of generated content (default: 100)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="test.txt",
        help="The name of the output file (default: test.txt)",
    )

    args = parser.parse_args()

    schedule = Scheduler(args.batch_size, args.generation_length, [])
    start_time = time.time()
    res = schedule.process(prompts)
    end_time = time.time()
    execution_time = end_time - start_time

    with open(args.output_file, "w", encoding="utf-8") as file:
        file.write("+" + "-" * 78 + "+\n")
        file.write(f"| {'Processed Prompts Collection':^76} |\n")
        file.write("+" + "=" * 78 + "+\n\n")

        for index, string in enumerate(res, start=1):
            file.write(f"Prompt {index}\n")
            file.write("-" * 80 + "\n")
            wrapped_text = textwrap.fill(string.strip(), width=76)
            for line in wrapped_text.split("\n"):
                file.write(f"| {line:<76} |\n")
            file.write("\n")

        file.write("+" + "=" * 78 + "+\n")
        file.write(f"| {'End of Document':^76} |\n")
        file.write("+" + "-" * 78 + "+\n")

    print(f"Batch Size: {args.batch_size}")
    print(f"Generation Length: {args.generation_length}")
    print(f"Number of processed prompts: {len(res)}")
    print(f"Execution Time: {execution_time:.4f} seconds")


if __name__ == "__main__":
    main()
