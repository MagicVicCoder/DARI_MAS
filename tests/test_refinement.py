import json
import unittest

from dari.hf_backend import output_value_spans
from dari.schema import Kind
from tests.test_runtime import make_runtime, pick
from tests import test_runtime as fixtures


class RefinementTests(unittest.TestCase):
    def test_draft_pooling_tracks_only_string_content(self):
        text = json.dumps({"work": "private", "draft": 'Compute x="1"; result 1.'})
        begin, end = output_value_spans(text)["draft"]
        self.assertEqual(text[begin:end], json.dumps('Compute x="1"; result 1.')[1:-1])

    def test_context_partitions_have_unique_question_and_exact_feature_spans(self):
        runtime = make_runtime()
        state = runtime.initial(fixtures.RuntimeTests().task())
        action = pick(runtime, state, Kind.HELP, width="narrow", quota="short")
        request = runtime.request_for_action(state, action)
        body = json.loads(request.prompt.split("\n", 1)[1])
        self.assertNotIn("packet", body)
        self.assertEqual(body["original_question_and_constraints"], state.task.question)
        self.assertEqual(set(request.feature_texts), set(request.feature_spans))
        for key, value in request.feature_texts.items():
            begin, end = request.feature_spans[key]
            self.assertEqual(request.prompt[begin:end], json.dumps(value, ensure_ascii=False)[1:-1])

    def test_draft_budget_counts_decoded_text_and_keeps_private_work_out(self):
        response = json.dumps({"work": "Long private reasoning " * 20, "draft": "1"})
        runtime = make_runtime(specialist_outputs=[response], draft_limit=1,
                               stage_output_budgets=(512,512), oneshot_output=512)
        state = runtime.initial(fixtures.RuntimeTests().task())
        state = runtime.step(state, pick(runtime, state, Kind.HELP, quota="short"))
        self.assertIsNotNone(state.help.draft)
        self.assertEqual(state.help.draft.text, "1")


if __name__ == "__main__":
    unittest.main()
