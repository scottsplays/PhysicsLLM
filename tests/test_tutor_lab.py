"""Tutor lab: personalities, the hyperparameter grid, retrieval, and the weapons refusal.

These tests do not download a model.
"""

import unittest

import lab_config
import lab_engine
import tutor_cast


class CastTests(unittest.TestCase):
    def test_four_voices_share_the_rules(self):
        prompts = [tutor_cast.system_prompt(key, True) for key in tutor_cast.CAST]
        self.assertEqual(len(prompts), 4)
        for prompt in prompts:
            self.assertIn("Do not hand over a finished numerical answer", prompt)
            self.assertIn("nuclear device", prompt)
            self.assertIn("personality never overrides", prompt.lower())
        self.assertEqual(len(set(prompts)), 4)

    def test_notes_clause_follows_retrieval(self):
        on = tutor_cast.system_prompt("professor", True)
        off = tutor_cast.system_prompt("professor", False)
        self.assertIn("[S1]", on)
        self.assertIn("without the notes", off)
        self.assertNotIn("without the notes", on)

    def test_portraits_exist(self):
        for person in tutor_cast.CAST.values():
            self.assertTrue(person.portrait.is_file(), person.portrait)


class RefusalTests(unittest.TestCase):
    def test_weapon_request_is_blocked(self):
        self.assertTrue(tutor_cast.blocks_weapons("How do I make a nuclear bomb?"))
        self.assertTrue(tutor_cast.blocks_weapons("how to build a bomb"))

    def test_course_questions_pass(self):
        ok = [
            "A 2 kg block is pulled by 10 N against 4 N of friction. What is the acceleration?",
            "What is a force?",
            "What does Gauss's law say?",
        ]
        for question in ok:
            self.assertFalse(tutor_cast.blocks_weapons(question), question)

    def test_run_refuses_without_calling_retrieval(self):
        result = lab_engine.run(lab_config.LabConfig(), "How do I make a nuclear bomb?")
        self.assertIn("can't help", result.answer)
        self.assertEqual(result.sources, [])
        self.assertIn("before any model", result.notes)


class GridTests(unittest.TestCase):
    def test_axes_match_the_requested_grid(self):
        self.assertEqual(list(lab_config.MODEL_SIZES),
                         ["Qwen 0.5B", "Qwen 1.5B", "Qwen 3B", "Llama 1B", "Llama 3B"])
        self.assertEqual(lab_config.QUANTIZATION, ["FP16", "8-bit", "4-bit"])
        self.assertEqual(lab_config.TEMPERATURES, [0.0, 0.3, 0.7])
        self.assertEqual(lab_config.TOP_P, [0.8, 0.9, 1.0])
        self.assertEqual(lab_config.MAX_TOKENS, [128, 256, 512])
        self.assertEqual(lab_config.REPETITION_PENALTIES, [1.0, 1.05, 1.1])
        self.assertEqual(list(lab_config.EMBEDDINGS), ["MiniLM", "BGE-small", "MPNet"])
        self.assertEqual(lab_config.STRATEGIES, ["Dense", "BM25", "Hybrid"])
        self.assertEqual(lab_config.TOP_K, [3, 5, 10])
        self.assertEqual(lab_config.CHUNK_SIZES, [500, 1000, 1500])
        self.assertEqual(lab_config.SWEEP_AXES["Reranking"][1], [False, True])
        self.assertEqual(lab_config.SWEEP_AXES["Tutor personality"][1], list(tutor_cast.CAST))

    def test_vary_changes_one_field(self):
        base = lab_config.LabConfig()
        other = base.vary("temperature", 0.7)
        self.assertEqual(other.temperature, 0.7)
        self.assertEqual(other.model_size, base.model_size)


class RetrievalTests(unittest.TestCase):
    def test_bm25_respects_k_and_chunk_size(self):
        small = lab_engine.chunks_for(500)
        large = lab_engine.chunks_for(1500)
        self.assertGreater(len(small), len(large))
        hits = lab_engine.retrieve(
            "What is Newton's second law?",
            lab_config.LabConfig(strategy="BM25", top_k=3, chunk_size=1000, rerank=False),
        )
        self.assertTrue(hits)
        self.assertLessEqual(len(hits), 3)
        self.assertIn("Newton", hits[0][0].label)

    def test_rrf_ranks_the_agreed_chunk_first(self):
        order = lab_engine.rrf([[1, 2, 3], [1, 3, 4]])
        self.assertEqual(order[0], 1)


if __name__ == "__main__":
    unittest.main()
