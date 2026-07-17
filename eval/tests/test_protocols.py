from __future__ import annotations

import json
import subprocess
from pathlib import Path

from eval.experiments.main_tables.matrix import expand_cells
from eval.experiments.main_tables.run import (
    expected_max_input_tokens,
    expected_max_new_tokens,
)


def test_paper_configs_and_manifests_are_json() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = {
        "table_5_6_main.json",
        "table_8_ood.json",
        "table_10_low_rank.json",
        "reported_scores.json",
    }
    config_dir = root / "eval" / "configs" / "paper"
    assert expected == {path.name for path in config_dir.glob("*.json")}
    for path in config_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("experiment") or payload.get("source")
    for path in (root / "eval").rglob("manifest*.json"):
        payload = path.read_text(encoding="utf-8")
        forbidden_roots = ("/" + "mnt/", "/" + "home/")
        assert not any(root in payload for root in forbidden_roots)
        json.loads(payload)
    for path in (root / "eval" / "experiments").rglob("configs/*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def test_original_ablation_protocol_fields_are_preserved() -> None:
    root = Path(__file__).resolve().parents[2]
    matrix = json.loads(
        (root / "eval/experiments/ablation/configs/ablation_matrix.json").read_text()
    )
    assert matrix["main_text_benchmark_ids"] == [
        "locomo_tps16",
        "nextmem_stm",
        "metis_test_nomixed",
        "metisops_v23_gold_turns",
    ]
    assert matrix["evaluation"] == {
        "context_policy": "memory_only",
        "query_style": "memory_direct",
        "dtype": "bfloat16",
        "max_new_tokens": 96,
        "judge_model": "gpt-4.1-mini",
        "judge_repeats": 3,
        "strict_only": True,
        "fail_on_judge_error": True,
        "fail_on_query_audit_issue": True,
        "memop_oom_policy": "fail",
    }


def test_low_rank_paper_grid_and_benchmarks_are_preserved() -> None:
    from eval.experiments.low_rank.run import BENCHMARKS, DEFAULT_RANKS

    assert DEFAULT_RANKS == [1, 4, 16, 64, 128, 256, "full"]
    assert [item.name for item in BENCHMARKS] == [
        "locomo_tps16",
        "nextmem_stm",
        "metis_test",
        "metisops_v23_gold_turns",
    ]


def test_release_matrices_have_the_audited_cell_counts() -> None:
    root = Path(__file__).resolve().parents[2]
    paper = root / "eval/configs/paper"
    main = json.loads((paper / "table_5_6_main.json").read_text())
    ood = json.loads((paper / "table_8_ood.json").read_text())
    low_rank = json.loads((paper / "table_10_low_rank.json").read_text())
    assert len(expand_cells(main, [], [])) == main["expected_cells"] == 77
    assert len(ood["methods"]) * len(ood["datasets"]) == ood["expected_cells"] == 14
    assert len(low_rank["models"]) * len(low_rank["ranks"]) * len(low_rank["benchmarks"]) == low_rank["expected_cells"] == 28
    assert "metis27b" in {item["id"] for item in ood["methods"]}
    for config in (main, ood):
        temp_lora27b = next(item for item in config["methods"] if item["id"] == "temp_lora27b")
        assert temp_lora27b["device_map"] == "balanced"
        assert temp_lora27b["max_memory"] == ["0:76GiB", "1:76GiB"]
    for method_id in (
        "qwen27b_no_context",
        "qwen27b_full_context",
        "dense_rag27b",
    ):
        method = next(item for item in main["methods"] if item["id"] == method_id)
        assert method["device_map"] == "auto"
        assert method["max_memory"] == ["0:75GiB", "1:75GiB"]
    assert {item["seed"] for item in main["methods"] if item["implementation"] == "temp_lora"} == {20260702}
    assert {item["seed"] for item in ood["methods"] if item["implementation"] == "temp_lora"} == {20260714}


def test_main_generation_caps_match_formal_launchers() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "eval/configs/paper/table_5_6_main.json").read_text()
    )
    assert expected_max_new_tokens(config, "full_context", "memops_gold") == 256
    assert expected_max_new_tokens(config, "partial_context", "metis_test") == 256
    assert expected_max_new_tokens(config, "full_context", "locomo_gold") == 96
    assert expected_max_new_tokens(config, "metis", "memops_gold") == 96
    assert expected_max_input_tokens(config, "full_context", "metis_test") == 32768
    assert expected_max_input_tokens(config, "full_context", "memops_gold") == 0


def test_temp_lora_27b_loading_is_forwarded_to_all_runners() -> None:
    root = Path(__file__).resolve().parents[2] / "eval"
    main = (root / "experiments/main_tables/run.py").read_text(encoding="utf-8")
    ood = (root / "experiments/ood/run.py").read_text(encoding="utf-8")
    runtime = (root / "methods/temp_lora/temp_lora_baseline.py").read_text(encoding="utf-8")
    assert "temp_lora_loading" in main
    assert 'args.method in {"metis", "temp_lora"}' in ood
    assert 'self.config.device_map in {"auto", "balanced"}' in runtime
    assert "self._set_seed()" in runtime
    assert 'env["PYTHONHASHSEED"] = str(args.seed)' in main
    assert 'env["PYTHONHASHSEED"] = str(args.seed)' in ood


def test_reported_score_axes_match_paper_protocols() -> None:
    root = Path(__file__).resolve().parents[2]
    paper = root / "eval/configs/paper"
    main = json.loads((paper / "table_5_6_main.json").read_text())
    ood = json.loads((paper / "table_8_ood.json").read_text())
    low_rank = json.loads((paper / "table_10_low_rank.json").read_text())
    reported = json.loads((paper / "reported_scores.json").read_text())
    assert set(reported["main"]["scores"]) == {item["id"] for item in main["methods"]}
    assert set(reported["ood"]["scores"]) == {item["id"] for item in ood["methods"]}
    assert reported["low_rank"]["ranks"] == low_rank["ranks"]
    assert set(reported["low_rank"]["scores"]) == {item["id"] for item in low_rank["benchmarks"]}


def test_single_data_manifest_matches_canonical_payload_size() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "eval/data/manifest.json").read_text())
    assert len(manifest["files"]) == 7
    assert sum(item["bytes"] for item in manifest["files"]) == 91_573_706


def test_ablation_manifest_excludes_audit_only_checkpoints() -> None:
    root = Path(__file__).resolve().parents[2]
    matrix = json.loads((root / "eval/experiments/ablation/configs/ablation_matrix.json").read_text())
    assert len(matrix["checkpoints"]) == 7
    assert {item["status"] for item in matrix["checkpoints"]} <= {"ready", "reference_ready"}


def test_ablation_checkpoints_use_the_shared_asset_registry() -> None:
    root = Path(__file__).resolve().parents[2]
    matrix = json.loads(
        (root / "eval/experiments/ablation/configs/ablation_matrix.json").read_text()
    )
    assets = json.loads((root / "eval/configs/assets.json").read_text())["assets"]
    assert {item["id"] for item in matrix["checkpoints"]} <= set(assets)


def test_release_text_has_no_private_server_paths_or_usernames() -> None:
    root = Path(__file__).resolve().parents[2]
    forbidden = (
        "/" + "mnt/afs/",
        "/" + "home/",
        "/" + "Users/",
        "api-" + "int.",
    )
    suffixes = {".py", ".json", ".md", ".toml", ".yml", ".txt"}
    for path in (root / "eval").rglob("*"):
        if path.is_file() and path.suffix in suffixes:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            assert not any(value.lower() in text for value in forbidden), path


def test_generated_result_metadata_is_git_ignored() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "eval/results/example/run_config.json",
        "eval/results/example/score_meta.json",
        "eval/results/example/audit.json",
        "eval/results/example/raw.jsonl",
    ):
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", relative],
            cwd=root,
            check=False,
        )
        assert completed.returncode == 0, relative
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "eval/results/README.md"],
        cwd=root,
        check=False,
    )
    assert completed.returncode == 1


def test_vendored_atm_metric_imports_without_external_checkout() -> None:
    from eval.benchmarks.ood.scripts import score_official_gold

    score_official_gold.load_atm_evaluator()
    assert score_official_gold.atm_number_core is not None
    assert score_official_gold.atm_list_core is not None


def test_unmigrated_paper_experiments_have_no_placeholder_runner() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("capacity", "general", "case_studies", "efficiency"):
        assert not (root / "eval/experiments" / name).exists()


def test_ood_score_command_propagates_smoke_limit() -> None:
    root = Path(__file__).resolve().parents[2] / "eval"
    source = (root / "experiments/ood/run.py").read_text(encoding="utf-8")
    scorer = (root / "benchmarks/ood/scripts/score_official_gold.py").read_text(encoding="utf-8")
    assert '"--limit"' in source and "str(args.limit)" in source
    assert 'parser.add_argument(\n        "--limit"' in scorer


def test_all_top_level_scorers_accept_openai_compatible_endpoint_options() -> None:
    from eval.experiments.ablation.run import build_parser as ablation_parser
    from eval.experiments.low_rank.matrix import build_parser as low_rank_parser

    for parser in (ablation_parser(), low_rank_parser()):
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        assert {"--judge-base-url", "--api-key-env"} <= options


def test_final_audits_reverify_data_and_judge_provenance() -> None:
    root = Path(__file__).resolve().parents[2] / "eval/experiments"
    for path in (
        root / "main_tables/audit.py",
        root / "ood/audit.py",
        root / "low_rank/audit.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "from eval.data.verify import verify" in source
        assert '"data": data_report' in source
        assert "judge_model" in source
    ood = (root / "ood/audit.py").read_text(encoding="utf-8")
    assert "scorer_code_sha256" in ood and "atm_revision" in ood
