import json
import unittest

from dari.protocol_json import repair_json
from dari.phase2b_evaluation import math_expression, score


class Phase2BTests(unittest.TestCase):
    def test_optional_runtime_repair_preserves_raw_call_and_strict_control(self):
        from tests.test_runtime import make_runtime, pick, RuntimeTests
        from dari.schema import Kind
        response='{"work":"Compute \\pi", "draft":"Use \\pi"}'
        runtime=make_runtime(specialist_outputs=[response],repair_protocol_json=True)
        state=runtime.initial(RuntimeTests().task())
        helped=runtime.step(state,pick(runtime,state,Kind.HELP,quota='short'))
        self.assertEqual(helped.help.draft.text,r'Use \pi')
        self.assertEqual(helped.events[-1].output,response)
        strict=make_runtime(specialist_outputs=[response])
        state=strict.initial(RuntimeTests().task())
        helped=strict.step(state,pick(strict,state,Kind.HELP,quota='short'))
        self.assertIsNone(helped.help.draft)

    def test_repair_preserves_literal_math_and_rejects_truncation(self):
        text = '{"work":"Use \\pi and \\sqrt{2}","draft":"result"}'
        fixed, changes = repair_json(text)
        self.assertEqual(json.loads(fixed)["work"], r"Use \pi and \sqrt{2}")
        self.assertTrue(changes)
        truncated = text[:-1]
        self.assertEqual(repair_json(truncated), (truncated, []))

    def test_valid_escapes_and_values_unchanged(self):
        text = json.dumps({"work":"line\nnext", "draft": "wrong answer 7.8"})
        self.assertEqual(repair_json(text), (text, []))
        fixed, _ = repair_json('```json\n'+text+'\n```')
        self.assertEqual(fixed,text)

    def test_single_final_math_span_without_reasoning_number_selection(self):
        self.assertEqual(math_expression(r'The simplified expression is $\frac{\cos x}{\sin x}$.'),r'\frac{\cos x}{\sin x}')
        self.assertEqual(score('math500',r'The simplified expression is $\frac{\cos x}{\sin x}$.',r'\cot x')['reward'],1)
        self.assertEqual(math_expression('Compare $2$ with $3$.'),'Compare $2$ with $3$.')
        self.assertEqual(math_expression('In 2 trials the value is $3$.'),'In 2 trials the value is $3$.')

    def test_local_result_terminal_punctuation_and_decimal_distinction(self):
        from dari.phase2b import local_assessment
        class Counter:
            def count_text_tokens(self, model_id, text):
                return len(text.split())
        row={'expected_local_result':'6496','analysis_category':3}
        output=json.dumps({'work':'6512-16=6496.','draft':'The area is 6496.'})
        self.assertTrue(local_assessment(row,output,Counter())['local_result_match'])
        output=json.dumps({'work':'6496.5','draft':'6496.5'})
        self.assertFalse(local_assessment(row,output,Counter())['local_result_match'])
