"""Ensemble of fine-tuned SILNLP models that share a target text, each reading its own source, on common test verses.

usage: python docs/research/prototype/exp13_model_ensemble.py --experiments <exp>,<exp>[,<exp>] --weights 0.3,0.5,0.7
           [--clearml-queue <queue> --task-name <project/name>] [--experiments-root <dir>] [--device cpu|cuda]
           [--dry-run]

Experiments are named as under MT/experiments.  The first is the base and each other one is ensembled with it.  Arms:
  single:<i>                    model i reading its own source
  ensemble:0+<i>:<rule>:<w>     the base model and model i, each reading its own source; w is the weight of model i
  one-model:0:source <i>        the base model alone reading the source of model i
  one-model:0:sources 0+<i>     the base model alone reading both sources, equal weights
  all:<rule>                    every model, equal weights
  check:...                     ensembles that must reproduce a single-model decode
Outputs and chrF3 scores go to a JSON file and, when this runs as a ClearML task, into the task's configuration
object "ensemble_outputs".
"""

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Imported before the arguments are parsed, so that ClearML records them and restores them on the worker.
import clearml  # noqa: F401
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from exp10_ensemble import Ensemble, Member  # noqa: E402


def file_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return text[:-1].split("\n") if text.endswith("\n") else text.split("\n")


class Experiment:
    def __init__(self, root: Path, name: str, checkpoint: str) -> None:
        self.name = name
        self._dir = root / name
        self.tokenizer = AutoTokenizer.from_pretrained(str(self._dir))
        self.verses = file_lines(self._dir / "test.vref.txt")
        pieces = file_lines(self._dir / "test.src.txt")
        references = file_lines(self._dir / "test.trg.detok.txt")
        if not len(self.verses) == len(pieces) == len(references) == len(set(self.verses)):
            raise RuntimeError(f"{name}: test files differ in length or repeat a verse reference")
        self._source = {v: self.tokenizer.convert_tokens_to_ids(line.split()) for v, line in zip(self.verses, pieces)}
        self._reference = dict(zip(self.verses, references))
        steps = sorted(int(p.name.split("-")[1]) for p in (self._dir / "run").glob("checkpoint-*"))
        self._step = checkpoint or str(steps[-1])
        saved = self._dir / f"test.trg-predictions.detok.txt.{self._step}"
        self._saved = dict(zip(self.verses, file_lines(saved))) if saved.is_file() else {}

    def source(self, verses: list[str]) -> list[list[int]]:
        return [self._source[v] for v in verses]

    def reference(self, verses: list[str]) -> list[str]:
        return [self._reference[v] for v in verses]

    def saved_predictions(self, verses: list[str]) -> list[str]:
        return [self._saved[v] for v in verses] if self._saved else []

    def load_model(self, device: str):
        checkpoint = self._dir / "run" / f"checkpoint-{self._step}"
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint))
        except Exception:
            print(f"loading from {checkpoint} failed, retrying from a local copy\n{traceback.format_exc()}", flush=True)
            local = self._plain_copy(checkpoint)
            model = AutoModelForSeq2SeqLM.from_pretrained(str(local))
            shutil.rmtree(local)
        model = model.eval().to(device)
        # SILNLP raises the limit to 512 for inference, whatever the checkpoint says.
        model.generation_config.max_length = max(model.generation_config.max_length or 0, 512)
        return model

    def _plain_copy(self, checkpoint: Path) -> Path:
        # Reads and writes only: on a network mount, copying metadata or mapping the file can fail where reads work.
        local = Path(tempfile.mkdtemp())
        for path in checkpoint.iterdir():
            if path.is_file():
                with path.open("rb") as source, (local / path.name).open("wb") as target:
                    shutil.copyfileobj(source, target, 64 * 2**20)
        return local


class ModelPool:
    """Keeps only the models the current arm needs in memory."""

    def __init__(self, experiments: list[Experiment], device: str) -> None:
        self._experiments = experiments
        self._device = device
        self._loaded: dict[int, object] = {}

    def models(self, indices: list[int]) -> list:
        for index in [i for i in self._loaded if i not in indices]:
            del self._loaded[index]
            gc.collect()
            if self._device.startswith("cuda"):
                torch.cuda.empty_cache()
        for index in indices:
            if index not in self._loaded:
                start = time.time()
                self._loaded[index] = self._experiments[index].load_model(self._device)
                print(f"loaded model {index} in {time.time() - start:.0f}s", flush=True)
        return [self._loaded[i] for i in indices]


class Decoder:
    def __init__(self, tokenizer, num_beams: int, batch_size: int, device: str) -> None:
        self._tokenizer = tokenizer
        self._num_beams = num_beams
        self._batch_size = batch_size
        self._device = device

    def decode(self, readers: list[tuple[object, list[list[int]]]], weights: list[float], rule: str) -> list[str]:
        # readers are (model, source token ids per verse); the first one drives beam search.
        outputs: list[str] = []
        for start in range(0, len(readers[0][1]), self._batch_size):
            batches = [self._padded(source[start : start + self._batch_size]) for _, source in readers]
            processors = LogitsProcessorList()
            if len(readers) > 1:
                members = [
                    Member(model, ids, mask, self._num_beams)
                    for (model, _), (ids, mask) in zip(readers[1:], batches[1:])
                ]
                processors.append(Ensemble(members, weights[0], weights[1:], rule))
            with torch.no_grad():
                generated = readers[0][0].generate(
                    input_ids=batches[0][0],
                    attention_mask=batches[0][1],
                    num_beams=self._num_beams,
                    logits_processor=processors,
                )
            for ids in generated.tolist():
                text = self._tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                outputs.append(self._tokenizer.clean_up_tokenization(text))
        return outputs

    def _padded(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(s) for s in sequences)
        pad = self._tokenizer.pad_token_id
        ids = torch.tensor([s + [pad] * (width - len(s)) for s in sequences], device=self._device)
        mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in sequences], device=self._device)
        return ids, mask


def chrf3(hypotheses: list[str], references: list[str]) -> float:
    # The call SILNLP's test script makes.
    return sacrebleu.corpus_chrf(hypotheses, [references], char_order=6, beta=3, remove_whitespace=True).score


def publish(task, results: dict) -> None:
    if task is None:
        return
    text = json.dumps(results, ensure_ascii=False)
    try:
        # Stored in the task record, which does not depend on the files server.
        task.set_configuration_object("ensemble_outputs", config_text=text)
    except Exception:
        print(f"could not store the outputs in the task, printing them instead\n{traceback.format_exc()}", flush=True)
        chunks = [text[i : i + 2000] for i in range(0, len(text), 2000)]
        for index, chunk in enumerate(chunks):
            print(f"OUTPUT-CHUNK {index + 1}/{len(chunks)} {chunk}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=str, default="")
    parser.add_argument("--weights", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--num-beams", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--experiments-root", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--out", type=str, default="ensemble_outputs.json")
    parser.add_argument("--clearml-queue", type=str, default="")
    parser.add_argument("--task-name", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    task = None
    if args.clearml_queue:
        from silnlp.nmt.clearml_connection import SILClearML

        task = SILClearML(args.task_name, args.clearml_queue, tag="research", skip_config=True).task

    names = [name for name in args.experiments.split(",") if name]
    if len(names) < 2:
        parser.error("--experiments needs at least two experiments")
    weights = [float(w) for w in args.weights.split(",")]
    root = Path(args.experiments_root or Path(os.environ["SIL_NLP_DATA_PATH"]) / "MT" / "experiments")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}, device {device}", flush=True)
    if device.startswith("cuda"):
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU {gpu.name}, {gpu.total_memory / 2**30:.0f} GB", flush=True)

    experiments = [Experiment(root, name, args.checkpoint) for name in names]
    base = experiments[0]
    vocabulary = base.tokenizer.get_vocab()
    for index, experiment in enumerate(experiments):
        same = experiment.tokenizer.get_vocab() == vocabulary
        print(f"experiment {index}: {experiment.name}, {len(experiment.verses)} test verses, same vocabulary: {same}")
        if not same:
            raise RuntimeError("the models do not share a vocabulary, so their next-token scores cannot be combined")
    verses = [v for v in base.verses if all(v in set(e.verses) for e in experiments)]
    print(f"{len(verses)} test verses are in every experiment's test set", flush=True)
    if args.dry_run or not verses:
        return

    references = base.reference(verses)
    sources = [e.source(verses) for e in experiments]
    pool = ModelPool(experiments, device)
    decoder = Decoder(base.tokenizer, args.num_beams, args.batch_size, device)
    results: dict = {"experiments": names, "verses": verses, "outputs": {}, "chrf3": {}, "checks": {}, "errors": {}}
    out_path = Path(args.out)

    def run(arm: str, indices: list[int], readers, arm_weights: list[float], rule: str = "linear") -> None:
        start = time.time()
        try:
            models = pool.models(indices)
            outputs = decoder.decode([(models[indices.index(i)], sources[s]) for i, s in readers], arm_weights, rule)
        except Exception:
            results["errors"][arm] = traceback.format_exc()
            print(f"FAILED {arm}\n{results['errors'][arm]}", flush=True)
            return
        results["outputs"][arm] = outputs
        results["chrf3"][arm] = chrf3(outputs, references)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{arm:40} chrF3 {results['chrf3'][arm]:6.2f}   {time.time() - start:4.0f}s", flush=True)

    def check(name: str, arm: str, expected_arm: str) -> None:
        got, want = results["outputs"].get(arm), results["outputs"].get(expected_arm)
        if got is not None and want is not None:
            results["checks"][name] = f"{sum(g == w for g, w in zip(got, want))}/{len(want)} identical"
            print(f"CHECK {name}: {results['checks'][name]}", flush=True)

    # readers are (model index, source index) pairs.
    for other in range(1, len(experiments)):
        pair = [0, other]
        if other == 1:
            run("single:0", pair, [(0, 0)], [1.0])
            run("check:0 twice", pair, [(0, 0), (0, 0)], [0.5, 0.5])
            check("the same model and source twice equals the single model", "check:0 twice", "single:0")
        run(f"single:{other}", pair, [(other, other)], [1.0])
        run(f"check:all weight on {other}", pair, [(0, 0), (other, other)], [0.0, 1.0])
        check(f"all weight on model {other} equals that model alone", f"check:all weight on {other}", f"single:{other}")
        for weight in weights:
            run(f"ensemble:0+{other}:linear:{weight:.2f}", pair, [(0, 0), (other, other)], [1.0 - weight, weight])
        run(f"ensemble:0+{other}:loglinear:0.50", pair, [(0, 0), (other, other)], [0.5, 0.5], "loglinear")
        run(f"one-model:0:source {other}", pair, [(0, other)], [1.0])
        run(f"one-model:0:sources 0+{other}", pair, [(0, 0), (0, other)], [0.5, 0.5])
        publish(task, results)
    if len(experiments) > 2:
        everyone = list(range(len(experiments)))
        run("all:linear", everyone, [(i, i) for i in everyone], [1.0 / len(everyone)] * len(everyone))

    for index, experiment in enumerate(experiments):
        saved, ours = experiment.saved_predictions(verses), results["outputs"].get(f"single:{index}")
        if saved and ours is not None:
            identical = sum(s == o for s, o in zip(saved, ours))
            results["checks"][
                f"single:{index} against the saved predictions"
            ] = f"{identical}/{len(saved)} identical, saved chrF3 {chrf3(saved, references):.2f}"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"chrf3": results["chrf3"], "checks": results["checks"]}, indent=1), flush=True)
    print(f"{len(results['errors'])} arms failed", flush=True)
    publish(task, results)


if __name__ == "__main__":
    main()
