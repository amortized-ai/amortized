"""Unit tests for the eval-set-mirrors-training-SDG guardrail.

DB-free: `_training_sdg_mirror_warning` builds `Repository(db)` internally, so
we monkeypatch that class with a fake that serves preset job rows by id. The
`_sdg_signature` helper is pure and tested directly.
"""

import pytest

from amortized.api import jobs


def _sdg_cfg(teacher: str, author_prompt: str, output_prompt: str) -> dict:
    """A minimal SDG config shaped like a real one (model_configs + llm columns)."""
    return {
        "num_records": 100,
        "model_configs": [{"alias": "teacher", "model": teacher}],
        "columns": [
            {"column_type": "sampler", "name": "topic"},
            {"column_type": "llm-text", "name": "input", "system_prompt": author_prompt},
            {"column_type": "llm-text", "name": "output", "system_prompt": output_prompt},
        ],
    }


class TestSdgSignature:
    def test_mirror_configs_compare_equal(self) -> None:
        # Same teacher + prompts; fresh inputs (seed/samples) don't enter the signature.
        a = _sdg_cfg("gpt-oss", "author", "assess RFE")
        b = _sdg_cfg("gpt-oss", "author", "assess RFE")
        b["num_records"] = 50  # held-out set is smaller — must not matter
        assert jobs._sdg_signature(a) == jobs._sdg_signature(b)

    def test_divergent_teacher_or_prompt_differs(self) -> None:
        base = _sdg_cfg("gpt-oss", "author", "assess RFE")
        assert jobs._sdg_signature(base) != jobs._sdg_signature(
            _sdg_cfg("gpt-4o-mini", "author", "assess RFE")
        )
        assert jobs._sdg_signature(base) != jobs._sdg_signature(
            _sdg_cfg("gpt-oss", "author", "prior batch rubric")
        )

    def test_different_topic_differs(self) -> None:
        # Same teacher + prompts but a different topic is a different task — the
        # signature must distinguish it (else a different-topic eval set is wrongly
        # flagged recipe_match / passes the mirror check).
        a = _sdg_cfg("gpt-oss", "author", "assess RFE")
        b = _sdg_cfg("gpt-oss", "author", "assess RFE")
        a["topic"] = "RFE tickets"
        b["topic"] = "support tickets"
        assert jobs._sdg_signature(a) != jobs._sdg_signature(b)

    def test_different_document_ids_differ(self) -> None:
        a = _sdg_cfg("gpt-oss", "author", "assess RFE")
        b = _sdg_cfg("gpt-oss", "author", "assess RFE")
        a["document_ids"] = ["doc-1", "doc-2"]
        b["document_ids"] = ["doc-9"]
        assert jobs._sdg_signature(a) != jobs._sdg_signature(b)
        # Order-insensitive: same set in a different order compares equal.
        c = _sdg_cfg("gpt-oss", "author", "assess RFE")
        c["document_ids"] = ["doc-2", "doc-1"]
        a2 = _sdg_cfg("gpt-oss", "author", "assess RFE")
        a2["document_ids"] = ["doc-1", "doc-2"]
        assert jobs._sdg_signature(a2) == jobs._sdg_signature(c)

    def test_accepts_json_string_config(self) -> None:
        import json

        cfg = _sdg_cfg("gpt-oss", "author", "assess RFE")
        assert jobs._sdg_signature(json.dumps(cfg)) == jobs._sdg_signature(cfg)

    def test_empty_or_bad_config_is_none(self) -> None:
        assert jobs._sdg_signature({}) is None
        assert jobs._sdg_signature("not json") is None
        assert jobs._sdg_signature(None) is None


class TestAssessorPrompt:
    def _cfg_with_processor(self, ref_col: str, assessor_prompt: str) -> dict:
        return {
            "columns": [
                {"column_type": "sampler", "name": "topic"},
                {"column_type": "llm-text", "name": "input", "system_prompt": "author"},
                {"column_type": "llm-text", "name": "score", "system_prompt": assessor_prompt},
            ],
            "processors": [
                {
                    "template": {
                        "messages": [
                            {"role": "user", "content": "{{ input }}"},
                            {"role": "assistant", "content": "{{ " + ref_col + " }}"},
                        ]
                    }
                }
            ],
        }

    def test_resolves_prompt_from_processor_assistant_ref(self) -> None:
        cfg = self._cfg_with_processor("score", "You are an RFE assessor.")
        assert jobs._assessor_prompt(cfg) == "You are an RFE assessor."

    def test_none_when_no_processor_ref(self) -> None:
        # No processor template -> we do NOT guess (no blind "last system_prompt"
        # fallback that could surface a sampler's prompt).
        cfg = {
            "columns": [
                {"name": "input", "system_prompt": "author"},
                {"name": "score", "system_prompt": "assessor"},
            ]
        }
        assert jobs._assessor_prompt(cfg) is None

    def test_none_on_malformed_config(self) -> None:
        assert jobs._assessor_prompt({}) is None
        assert jobs._assessor_prompt({"columns": "nope"}) is None


class _FakeRepo:
    def __init__(self, jobs_by_id: dict) -> None:
        self._jobs = jobs_by_id

    async def get_job(self, job_id: str):
        return self._jobs.get(job_id)


@pytest.fixture
def patch_repo(monkeypatch):
    def _install(jobs_by_id: dict) -> None:
        monkeypatch.setattr(jobs, "Repository", lambda _db: _FakeRepo(jobs_by_id))

    return _install


class TestTrainingSdgMirrorWarning:
    @pytest.mark.asyncio
    async def test_warns_on_divergent_teacher_and_prompt(self, patch_repo) -> None:
        patch_repo(
            {
                "train-job": {"parent_job_id": "train-sdg"},
                "train-sdg": {"config": _sdg_cfg("gpt-oss", "author", "assess RFE")},
                "eval-sdg": {"config": _sdg_cfg("gpt-4o-mini", "author", "prior batch")},
            }
        )
        warnings = await jobs._training_sdg_mirror_warning(
            {"training_job_id": "train-job"}, "eval-sdg", db=None
        )
        assert warnings and "NOT generated by the same SDG" in warnings[0]
        assert "teacher model" in warnings[0]
        assert "assessor/system prompt" in warnings[0]
        assert "train-sdg" in warnings[0]  # points at the config to reuse

    @pytest.mark.asyncio
    async def test_warns_and_names_topic_doc_column_divergence(self, patch_repo) -> None:
        # Same teacher + prompts but a different topic/documents/columns: the signature
        # diverges, so the warning must fire AND name the component — never an empty
        # "differs in: )" (the diff builder used to cover only teacher + prompt).
        train = _sdg_cfg("gpt-oss", "author", "assess RFE")
        train["topic"] = "support tickets"
        train["document_ids"] = ["doc-1"]
        eval_cfg = _sdg_cfg("gpt-oss", "author", "assess RFE")
        eval_cfg["topic"] = "sales emails"
        eval_cfg["document_ids"] = ["doc-9"]
        patch_repo(
            {
                "train-job": {"parent_job_id": "train-sdg"},
                "train-sdg": {"config": train},
                "eval-sdg": {"config": eval_cfg},
            }
        )
        warnings = await jobs._training_sdg_mirror_warning(
            {"training_job_id": "train-job"}, "eval-sdg", db=None
        )
        assert warnings
        assert "differs in: )" not in warnings[0]  # never an empty reason
        assert "topic" in warnings[0]
        assert "source documents" in warnings[0]

    @pytest.mark.asyncio
    async def test_silent_when_eval_set_mirrors_training(self, patch_repo) -> None:
        cfg = _sdg_cfg("gpt-oss", "author", "assess RFE")
        patch_repo(
            {
                "train-job": {"parent_job_id": "train-sdg"},
                "train-sdg": {"config": cfg},
                # fresh inputs only (same teacher + prompts) -> signatures match
                "eval-sdg": {"config": {**cfg, "num_records": 40}},
            }
        )
        warnings = await jobs._training_sdg_mirror_warning(
            {"training_job_id": "train-job"}, "eval-sdg", db=None
        )
        assert warnings == []

    @pytest.mark.asyncio
    async def test_silent_when_not_for_a_trained_model(self, patch_repo) -> None:
        patch_repo({})  # no training_job_id -> never resolves lineage
        warnings = await jobs._training_sdg_mirror_warning(
            {}, "eval-sdg", db=None
        )
        assert warnings == []

    @pytest.mark.asyncio
    async def test_silent_when_training_not_chained_from_sdg(self, patch_repo) -> None:
        # Trained from a split/upload: no parent SDG to mirror -> best-effort silent.
        patch_repo({"train-job": {"parent_job_id": None}})
        warnings = await jobs._training_sdg_mirror_warning(
            {"training_job_id": "train-job"}, "eval-sdg", db=None
        )
        assert warnings == []

    @pytest.mark.asyncio
    async def test_silent_when_eval_data_not_a_chained_sdg(self, patch_repo) -> None:
        # Eval data is an uploaded dataset / split (no parent_job_id) -> can't compare.
        patch_repo({"train-job": {"parent_job_id": "train-sdg"}})
        warnings = await jobs._training_sdg_mirror_warning(
            {"training_job_id": "train-job"}, "", db=None
        )
        assert warnings == []


# ---------------------------------------------------------------------------
# clone_sdg_config_for_eval — deterministic eval-SDG mirror
# ---------------------------------------------------------------------------


class TestCloneSdgConfigForEval:
    @pytest.mark.asyncio
    async def test_mirrors_recipe_and_sets_record_count(self, patch_repo) -> None:
        cfg = _sdg_cfg("gpt-oss", "author", "assess RFE")
        cfg["topic"] = "support tickets"
        cfg["processors"] = [{"processor_type": "schema_transform"}]
        patch_repo(
            {
                "train-job": {"parent_job_id": "train-sdg"},
                "train-sdg": {"config": cfg},
            }
        )
        req = jobs.CloneSdgForEvalRequest(training_job_id="train-job", num_records=40)
        result = await jobs.clone_sdg_config_for_eval(req, db=None)

        assert result.train_sdg_job_id == "train-sdg"
        assert result.config["num_records"] == 40  # user-chosen eval size, not 100
        # teacher + prompts + format copied verbatim -> same signature as training
        assert jobs._sdg_signature(result.config) == jobs._sdg_signature(cfg)
        assert result.config["topic"] == "support tickets"
        assert result.config["processors"] == cfg["processors"]

    @pytest.mark.asyncio
    async def test_carries_constraints_tool_configs_drops_non_recipe(
        self, patch_repo
    ) -> None:
        # Full-recipe copy: recipe-defining fields the old allowlist dropped
        # (constraints, tool_configs) must carry over, while non-recipe fields
        # (parent_job_id lineage, mode) must NOT.
        cfg = _sdg_cfg("gpt-oss", "author", "assess RFE")
        cfg["constraints"] = [{"target_column": "output", "type": "non_empty"}]
        cfg["tool_configs"] = [{"name": "web_search"}]
        cfg["parent_job_id"] = "some-upstream"
        cfg["mode"] = "preview"
        patch_repo(
            {
                "train-job": {"parent_job_id": "train-sdg"},
                "train-sdg": {"config": cfg},
            }
        )
        req = jobs.CloneSdgForEvalRequest(training_job_id="train-job", num_records=40)
        result = await jobs.clone_sdg_config_for_eval(req, db=None)
        assert result.config["constraints"] == cfg["constraints"]
        assert result.config["tool_configs"] == cfg["tool_configs"]
        assert "parent_job_id" not in result.config
        assert "mode" not in result.config
        assert result.config["num_records"] == 40

    @pytest.mark.asyncio
    async def test_synthetic_recipe_note_claims_held_out(self, patch_repo) -> None:
        # No document_ids -> fresh generation genuinely is held-out; say so.
        cfg = _sdg_cfg("gpt-oss", "author", "assess RFE")
        patch_repo(
            {
                "train-job": {"parent_job_id": "train-sdg"},
                "train-sdg": {"config": cfg},
            }
        )
        req = jobs.CloneSdgForEvalRequest(training_job_id="train-job", num_records=40)
        result = await jobs.clone_sdg_config_for_eval(req, db=None)
        assert "fresh held-out inputs" in result.note
        assert "disjoint" not in result.note

    @pytest.mark.asyncio
    async def test_document_grounded_note_disclaims_held_out(self, patch_repo) -> None:
        # document_ids mirrored -> eval re-seeds from the SAME chunks as training,
        # so the note must NOT claim a disjoint held-out split (leakage honesty).
        cfg = _sdg_cfg("gpt-oss", "author", "assess RFE")
        cfg["document_ids"] = ["doc-1", "doc-2"]
        patch_repo(
            {
                "train-job": {"parent_job_id": "train-sdg"},
                "train-sdg": {"config": cfg},
            }
        )
        req = jobs.CloneSdgForEvalRequest(training_job_id="train-job", num_records=40)
        result = await jobs.clone_sdg_config_for_eval(req, db=None)
        assert result.config["document_ids"] == ["doc-1", "doc-2"]  # still mirrored
        assert "NOT a disjoint held-out split" in result.note
        assert "fresh held-out inputs" not in result.note

    @pytest.mark.asyncio
    async def test_accepts_json_string_config(self, patch_repo) -> None:
        import json

        cfg = _sdg_cfg("gpt-oss", "author", "assess RFE")
        patch_repo(
            {
                "train-job": {"parent_job_id": "train-sdg"},
                "train-sdg": {"config": json.dumps(cfg)},
            }
        )
        req = jobs.CloneSdgForEvalRequest(training_job_id="train-job", num_records=25)
        result = await jobs.clone_sdg_config_for_eval(req, db=None)
        assert result.config["num_records"] == 25
        assert jobs._sdg_signature(result.config) == jobs._sdg_signature(cfg)

    @pytest.mark.asyncio
    async def test_404_when_training_job_missing(self, patch_repo) -> None:
        from fastapi import HTTPException

        patch_repo({})
        req = jobs.CloneSdgForEvalRequest(training_job_id="nope", num_records=40)
        with pytest.raises(HTTPException) as exc:
            await jobs.clone_sdg_config_for_eval(req, db=None)
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_422_when_training_has_no_parent_sdg(self, patch_repo) -> None:
        from fastapi import HTTPException

        patch_repo({"train-job": {"parent_job_id": None}})
        req = jobs.CloneSdgForEvalRequest(training_job_id="train-job", num_records=40)
        with pytest.raises(HTTPException) as exc:
            await jobs.clone_sdg_config_for_eval(req, db=None)
        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_422_when_parent_sdg_config_empty(self, patch_repo) -> None:
        from fastapi import HTTPException

        patch_repo(
            {
                "train-job": {"parent_job_id": "train-sdg"},
                "train-sdg": {"config": {}},
            }
        )
        req = jobs.CloneSdgForEvalRequest(training_job_id="train-job", num_records=40)
        with pytest.raises(HTTPException) as exc:
            await jobs.clone_sdg_config_for_eval(req, db=None)
        assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# Eval-vs-training record overlap (leakage) guardrail
# ---------------------------------------------------------------------------


def _msg_rec(user: str, answer: str) -> dict:
    """An SFT `messages` record — the shape SDG datasets use."""
    return {
        "messages": [
            {"role": "system", "content": "assess the ticket"},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]
    }


class TestRecordInputSignature:
    def test_keys_on_user_content_not_the_answer(self) -> None:
        # Same input, different generated answer -> same signature (still leakage).
        assert jobs._record_input_signature(
            _msg_rec("ticket A", "9")
        ) == jobs._record_input_signature(_msg_rec("ticket A", "7"))

    def test_different_input_differs(self) -> None:
        assert jobs._record_input_signature(
            _msg_rec("ticket A", "9")
        ) != jobs._record_input_signature(_msg_rec("ticket B", "9"))

    def test_falls_back_to_whole_record_without_messages(self) -> None:
        a = jobs._record_input_signature({"input": "x", "label": "p"})
        b = jobs._record_input_signature({"input": "x", "label": "q"})
        assert a != b  # no messages column -> whole-record hash distinguishes them


class TestEvalOverlapWarning:
    @pytest.fixture
    def patch(self, monkeypatch):
        def _install(jobs_by_id: dict, sigs_by_run: dict) -> None:
            monkeypatch.setattr(jobs, "Repository", lambda _db: _FakeRepo(jobs_by_id))

            async def _fake_sigs(run_id: str):
                return sigs_by_run[run_id]

            monkeypatch.setattr(jobs, "_input_signatures", _fake_sigs)

        return _install

    @pytest.mark.asyncio
    async def test_warns_on_reused_inputs(self, patch) -> None:
        patch(
            {
                "train-job": {"config": {}, "parent_job_id": "train-sdg"},
                "train-sdg": {"mlflow_run_id": "run-train"},
                "eval-sdg": {"mlflow_run_id": "run-eval"},
            },
            {"run-train": {"a", "b"}, "run-eval": {"a", "c"}},  # 'a' reused
        )
        warnings = await jobs._eval_overlap_warning(
            {"training_job_id": "train-job"}, "eval-sdg", db=None
        )
        assert warnings and "1 of 2 eval records reuse inputs" in warnings[0]

    @pytest.mark.asyncio
    async def test_silent_on_fresh_eval_set(self, patch) -> None:
        patch(
            {
                "train-job": {"config": {}, "parent_job_id": "train-sdg"},
                "train-sdg": {"mlflow_run_id": "run-train"},
                "eval-sdg": {"mlflow_run_id": "run-eval"},
            },
            {"run-train": {"a", "b"}, "run-eval": {"c", "d"}},
        )
        warnings = await jobs._eval_overlap_warning(
            {"training_job_id": "train-job"}, "eval-sdg", db=None
        )
        assert warnings == []

    @pytest.mark.asyncio
    async def test_flags_eval_on_the_training_dataset_itself(self, patch) -> None:
        # Eval data resolves to the SAME run as training -> total leakage.
        patch(
            {
                "train-job": {"config": {"data_run_id": "run-shared"}},
                "eval-sdg": {"mlflow_run_id": "run-shared"},
            },
            {"run-shared": {"a", "b"}},
        )
        warnings = await jobs._eval_overlap_warning(
            {"training_job_id": "train-job"}, "eval-sdg", db=None
        )
        assert warnings and "is the model's training dataset" in warnings[0]

    @pytest.mark.asyncio
    async def test_prefers_training_data_run_id_over_parent(self, patch) -> None:
        # Model trained on a split complement (data_run_id), not the full SDG.
        patch(
            {
                "train-job": {
                    "config": {"data_run_id": "run-complement"},
                    "parent_job_id": "train-sdg",
                },
                "eval-sdg": {"mlflow_run_id": "run-eval"},
            },
            {"run-complement": {"a"}, "run-eval": {"a", "z"}},  # overlaps complement
        )
        warnings = await jobs._eval_overlap_warning(
            {"training_job_id": "train-job"}, "eval-sdg", db=None
        )
        assert warnings and "1 of 2 eval records reuse inputs" in warnings[0]

    @pytest.mark.asyncio
    async def test_silent_when_not_for_a_trained_model(self, patch) -> None:
        patch({}, {})
        assert await jobs._eval_overlap_warning({}, "eval-sdg", db=None) == []

    @pytest.mark.asyncio
    async def test_silent_when_dataset_unresolvable(self, patch) -> None:
        patch({"train-job": {"config": {}, "parent_job_id": "missing"}}, {})
        warnings = await jobs._eval_overlap_warning(
            {"training_job_id": "train-job"}, "eval-sdg", db=None
        )
        assert warnings == []

    @pytest.mark.asyncio
    async def test_silent_on_dataset_load_error(self, monkeypatch) -> None:
        monkeypatch.setattr(
            jobs,
            "Repository",
            lambda _db: _FakeRepo(
                {
                    "train-job": {"config": {}, "parent_job_id": "train-sdg"},
                    "train-sdg": {"mlflow_run_id": "run-train"},
                    "eval-sdg": {"mlflow_run_id": "run-eval"},
                }
            ),
        )

        async def _boom(run_id: str):
            raise RuntimeError("mlflow down")

        monkeypatch.setattr(jobs, "_input_signatures", _boom)
        warnings = await jobs._eval_overlap_warning(
            {"training_job_id": "train-job"}, "eval-sdg", db=None
        )
        assert warnings == []
