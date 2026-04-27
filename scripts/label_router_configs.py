import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from types import SimpleNamespace

from datasets import load_dataset, load_from_disk
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)

from main import prepare_dataset  # noqa: E402
from src.evaluate_results import eval_acc, eval_f1, eval_hit, normalize  # noqa: E402
from src.llm_navigator import LLM_Navigator  # noqa: E402


MAX_LENGTHS = [1, 2, 3, 4]
TOP_KS = [1, 2, 3, 5]
BOOLEAN_ANSWERS = {"true", "false"}


def config_grid():
    configs = [
        {"max_length": max_length, "top_k": top_k, "cost": max_length * top_k}
        for max_length in MAX_LENGTHS
        for top_k in TOP_KS
    ]
    return sorted(configs, key=lambda item: (item["cost"], item["max_length"], item["top_k"]))


def prepare_crlt_sample(sample):
    ground_paths = []
    for step in sample["reasoning_steps"]:
        facts = step.get("facts used in this step")
        if facts is None:
            continue
        if isinstance(facts, list):
            ground_paths.extend(facts)
        else:
            ground_paths.append(facts)

    sample["ground_paths"] = ground_paths
    return sample


def load_seen_ids(output_file, rerun_unresolved=False):
    seen = set()
    if not os.path.exists(output_file):
        return seen

    with open(output_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if rerun_unresolved and item.get("status") == "unresolved":
                continue
            sample_id = item.get("id")
            if sample_id is not None:
                seen.add(sample_id)
    return seen


def load_train_dataset(args):
    cache_name = f"{args.benchmark}_train_processed"
    cached_dataset_path = os.path.join(args.save_cache, cache_name)
    if os.path.exists(cached_dataset_path):
        dataset = load_from_disk(cached_dataset_path)
        if args.benchmark != "CL-LT-KGQA":
            return dataset
    else:
        if args.benchmark in ["RoG-webqsp", "RoG-cwq"]:
            input_file = os.path.join(args.data_path, args.benchmark)
            dataset = load_dataset(input_file, split="train", cache_dir=args.save_cache)
            dataset = dataset.map(prepare_dataset, num_proc=args.N_CPUS if args.N_CPUS > 1 else None)
        elif args.benchmark == "CL-LT-KGQA":
            input_file = os.path.join(args.crlt_data_dir, "CR-LT-QA-Wikidata-Cache.jsonl")
            dataset = load_dataset("json", data_files=input_file, split="train", cache_dir=args.save_cache)
            dataset = dataset.map(prepare_crlt_sample, num_proc=args.N_CPUS if args.N_CPUS > 1 else None)
        else:
            raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    dataset = dataset.filter(
        lambda x: x.get("hop") > 0
        and x.get("question") != ""
        and len(x.get("q_entity")) > 0
        and len(x.get("a_entity")) > 0
        and len(x.get("ground_paths")) > 0,
        num_proc=args.N_CPUS if args.N_CPUS > 1 else None,
    )
    dataset = dataset.filter(
        lambda x: x.get("q_entity") is not None,
        num_proc=args.N_CPUS if args.N_CPUS > 1 else None,
    )
    if args.benchmark == "CL-LT-KGQA":
        dataset = dataset.filter(
            lambda x: is_boolean_answer(x.get("a_entity")),
            num_proc=args.N_CPUS if args.N_CPUS > 1 else None,
        )

    if not os.path.exists(cached_dataset_path):
        os.makedirs(cached_dataset_path, exist_ok=True)
        dataset.save_to_disk(cached_dataset_path)
    return dataset


def make_fidelis_args(args, max_length, top_k):
    return SimpleNamespace(
        N_CPUS=args.N_CPUS,
        sample=-1,
        data_path=args.data_path,
        d=args.benchmark,
        save_cache=args.save_cache,
        split="train",
        output_path=args.output_path,
        model_name=args.model_name,
        top_n=args.top_n,
        top_k=top_k,
        max_length=max_length,
        strategy=args.strategy,
        squeeze=True,
        verifier=args.verifier,
        embedding_model=args.embedding_model,
        add_hop_information=args.add_hop_information,
        generate_embeddings=False,
        alpha=args.alpha,
        debug=args.debug,
    )


def split_predictions(prediction):
    if prediction is None:
        return []
    if isinstance(prediction, list):
        return prediction
    return [item for item in prediction.split("\n") if item.strip()]


def is_boolean_answer(answer):
    if not isinstance(answer, list) or len(answer) != 1:
        return False
    return normalize(str(answer[0])) in BOOLEAN_ANSWERS


def extract_boolean_prediction(prediction):
    values = []
    for item in split_predictions(prediction):
        normalized = normalize(str(item))
        if normalized in BOOLEAN_ANSWERS:
            values.append(normalized)

    unique_values = set(values)
    if len(unique_values) == 1:
        return values[0]
    return None


def score_prediction(prediction, answer):
    prediction_list = split_predictions(prediction)
    f1, precision, recall = eval_f1(prediction_list, answer)
    prediction_str = " ".join(prediction_list)
    return {
        "acc": eval_acc(prediction_str, answer),
        "hit": eval_hit(prediction_str, answer),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def score_boolean_prediction(prediction, answer):
    gold = normalize(str(answer[0])) if is_boolean_answer(answer) else None
    predicted = extract_boolean_prediction(prediction)
    is_correct = gold is not None and predicted == gold
    score = 1.0 if is_correct else 0.0
    return {
        "acc": score,
        "hit": int(is_correct),
        "f1": score,
        "precision": score,
        "recall": score,
        "prediction_boolean": predicted,
        "ground_truth_boolean": gold,
    }


def score_for_benchmark(prediction, answer, benchmark):
    if benchmark == "CL-LT-KGQA" and is_boolean_answer(answer):
        return score_boolean_prediction(prediction, answer)
    return score_prediction(prediction, answer)


def run_one_config(sample, args, config, navigators):
    key = (config["max_length"], config["top_k"])
    if key not in navigators:
        fidelis_args = make_fidelis_args(args, config["max_length"], config["top_k"])
        navigators[key] = LLM_Navigator(fidelis_args)

    started_at = time.time()
    res, _ = navigators[key].beam_search(sample)
    llm_metrics = score_for_benchmark(res.get("prediction_llm", ""), res["ground_truth"], args.benchmark)
    direct_metrics = score_for_benchmark(
        res.get("prediction_direct_answer", ""), res["ground_truth"], args.benchmark
    )

    return {
        "config": copy.deepcopy(config),
        "runtime_sec": round(time.time() - started_at, 3),
        "is_correct": llm_metrics["f1"] == 1.0,
        "prediction_llm": res.get("prediction_llm", ""),
        "prediction_direct_answer": res.get("prediction_direct_answer", ""),
        "reasoning_path": res.get("reasoning_path", []),
        "metrics": {
            "prediction_llm": llm_metrics,
            "prediction_direct_answer": direct_metrics,
        },
    }


def append_jsonl(path, item):
    with open(path, "a") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


def label_samples(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = args.output_file or os.path.join(
        args.output_dir, f"{args.benchmark}_train_router_labels.jsonl"
    )
    error_file = output_file.replace(".jsonl", "_errors.jsonl")

    if args.num_samples <= 0:
        print(f"No labels requested. Output file would be {output_file}")
        return

    seen_ids = load_seen_ids(output_file, rerun_unresolved=args.rerun_unresolved)
    dataset = load_train_dataset(args)
    if args.random_sample:
        sample_ids = dataset["id"]
        sample_indices = [idx for idx, sample_id in enumerate(sample_ids) if sample_id not in seen_ids]
        rng = random.Random(args.seed)
        rng.shuffle(sample_indices)
        if args.start_index:
            sample_indices = sample_indices[args.start_index:]
    else:
        sample_indices = list(range(len(dataset)))

    configs = config_grid()
    navigators = {}

    labeled_count = 0
    scanned_count = 0
    pbar = tqdm(total=args.num_samples, desc="Labeling router configs")

    for sample_idx in sample_indices:
        sample = dataset[sample_idx]
        sample_id = sample.get("id")
        if sample_id in seen_ids:
            continue
        if args.start_index and scanned_count < args.start_index:
            scanned_count += 1
            continue

        scanned_count += 1
        attempts = []
        label = None
        status = "unresolved"

        for config in configs:
            try:
                attempt = run_one_config(sample, args, config, navigators)
                attempts.append(attempt)
            except Exception as e:
                error_item = {
                    "benchmark": args.benchmark,
                    "split": "train",
                    "id": sample_id,
                    "question": sample.get("question"),
                    "config": config,
                    "error": str(e),
                }
                append_jsonl(error_file, error_item)
                attempts.append({
                    "config": copy.deepcopy(config),
                    "is_correct": False,
                    "error": str(e),
                })
                if args.stop_on_error:
                    raise
                continue

            if attempt["is_correct"]:
                label = {
                    "max_length": config["max_length"],
                    "top_k": config["top_k"],
                    "cost": config["cost"],
                }
                status = "resolved"
                break

        record = {
            "benchmark": args.benchmark,
            "split": "train",
            "id": sample_id,
            "question": sample.get("question"),
            "q_entities": sample.get("q_entity"),
            "ground_truth": sample.get("a_entity"),
            "hop": sample.get("hop"),
            "status": status,
            "label": label,
            "attempts": attempts,
        }
        append_jsonl(output_file, record)
        seen_ids.add(sample_id)
        labeled_count += 1
        pbar.update(1)

        if labeled_count >= args.num_samples:
            break

    pbar.close()
    print(f"Saved {labeled_count} new labels to {output_file}")
    if labeled_count < args.num_samples:
        print(f"Only found {labeled_count} unlabeled samples before the dataset ended.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Label router configs by running FiDeLiS on train split with max_length/top_k grid."
    )
    parser.add_argument(
        "--benchmark",
        "-d",
        choices=["RoG-webqsp", "RoG-cwq", "CL-LT-KGQA"],
        default="RoG-webqsp",
    )
    parser.add_argument("--num_samples", "--sample", type=int, default=10)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="rmanluo")
    parser.add_argument("--crlt_data_dir", type=str, default="datasets/crlt")
    parser.add_argument("--save_cache", type=str, default="cache")
    parser.add_argument("--output_path", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="router_labels")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--strategy", type=str, default="discrete_rating")
    parser.add_argument("--verifier", type=str, default="deductive+planning")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--add_hop_information", action="store_true")
    parser.add_argument("--N_CPUS", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rerun_unresolved", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    os.environ.setdefault("WANDB_MODE", "disabled")
    label_samples(parse_args())
