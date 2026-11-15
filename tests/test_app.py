"""Run with .venv/Scripts/python.exe -m unittest discover -s tests."""
import math
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from tensor import Vector, Matrix
from layer import Layer
from model import Model
from copy import deepcopy
from backprop_inspector import backward_snapshot
from iris_app import forward_snapshot
from visualization_helpers import display_values
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]

class AppTests(unittest.TestCase):
    def test_conversions_and_cache(self):
        for value in [Vector([1, 2]), Layer([1, 2])]:
            converted = value.to_list()
            converted[0] = 99
            self.assertEqual(value.to_list(), [1., 2.])
        matrix = Matrix([[1, 2], [3, 4]])
        matrix.to_list()[0][0] = 99
        self.assertEqual(matrix.to_list()[0][0], 1)
        model = Model(2, 3, 2)
        before = [w.weight_matrix.to_list() for w in model.weights]
        probs, steps = forward_snapshot(model, [0.1, -0.2])
        self.assertAlmostEqual(sum(probs), 1)
        for i, step in enumerate(steps):
            expected = [sum(x * row[j] for x, row in zip(step['xb'], step['W']))
                        for j in range(len(step['z']))]
            for a, b in zip(expected, step['z']):
                self.assertAlmostEqual(a, b)
            self.assertEqual(step['y'], [max(0, v) for v in step['z']] if i < 2 else step['z'])
        self.assertEqual(before, [w.weight_matrix.to_list() for w in model.weights])

    def test_app_datasets_and_reruns(self):
        for dataset in ['Iris', 'Wine', 'Moons']:
            with self.subTest(dataset=dataset):
                at = AppTest.from_file(str(ROOT / 'iris_app.py'), default_timeout=120).run()
                self.assertFalse(at.exception)
                at.selectbox[0].set_value(dataset).run()
                at.number_input[0].set_value(2)
                at.button[0].click().run()
                self.assertFalse(at.exception)
                self.assertFalse(at.error)
                result = at.session_state['result']
                self.assertIsInstance(result['model'], Model)
                self.assertEqual(len(result['history']), 2)
                self.assertEqual(len(result['train_accuracy']), 2)
                self.assertEqual(len(result['validation_accuracy']), 2)
                self.assertAlmostEqual(result['validation_accuracy'][-1], accuracy_score(result['data']['y_test'], result['test_pred']))
                self.assertTrue(all(0 <= v <= 1 for v in result['train_accuracy'] + result['validation_accuracy']))
                self.assertIn('Math Inspector', [t.label for t in at.tabs])
                self.assertIn('Backprop Inspector', [t.label for t in at.tabs])
                self.assertTrue(all(math.isfinite(v) for v in result['history']))
                before = [w.weight_matrix.to_list() for w in result['model'].weights]
                next(s for s in at.slider if s.label == 'Held-out sample').set_value(1).run()
                self.assertFalse(at.exception)
                self.assertEqual(before, [w.weight_matrix.to_list() for w in at.session_state['result']['model'].weights])
                next(s for s in at.slider if s.label == 'Backprop sample').set_value(1).run()
                self.assertFalse(at.exception)
                self.assertEqual(before, [w.weight_matrix.to_list() for w in at.session_state['result']['model'].weights])
                next(s for s in at.slider if s.label == 'Hidden size (both layers)').set_value(4).run()
                self.assertFalse(at.exception)
                self.assertTrue(any('Settings changed' in i.value for i in at.info))
                at.button[0].click().run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state['result']['model'].fc1.weight_matrix.shape[1], 4)

    def test_backprop_matches_real_update(self):
        model = Model(2, 3, 2)
        sample, target = [0.1, -0.2], [1., 0.]
        model.forward(sample)
        before = deepcopy(model)
        loss, gradients = backward_snapshot(model, sample, target)
        actual = deepcopy(model)
        self.assertAlmostEqual(loss, actual.backward(sample, target, 0.01))
        for index, g in zip([2, 1, 0], gradients):
            original = before.weights[index]
            self.assertEqual(model.weights[index].weight_matrix.to_list(), original.weight_matrix.to_list())
            self.assertEqual(model.weights[index].z.to_list(), original.z.to_list())
            for row in range(original.weight_matrix.n):
                for col in range(original.weight_matrix.m):
                    self.assertAlmostEqual(actual.weights[index].weight_matrix[row][col],
                        original.weight_matrix[row][col] - 0.01 * g['dW'][row][col])
        for upper, lower in zip(gradients, gradients[1:]):
            self.assertEqual(upper['dx'].tolist(), lower['dy'].tolist())
        # Finite differences verify the displayed gradient of one fc3 parameter.
        plus, minus = deepcopy(model), deepcopy(model)
        eps = 1e-5
        for copy, delta in [(plus, eps), (minus, -eps)]:
            values = copy.fc3.weight_matrix.to_list()
            values[0][0] += delta
            copy.fc3.weight_matrix = Matrix(values)
        import mathematics
        lp = mathematics.CrossEntropyLoss(target, plus.forward(sample).to_list())
        lm = mathematics.CrossEntropyLoss(target, minus.forward(sample).to_list())
        self.assertAlmostEqual((lp-lm)/(2*eps), gradients[0]['dW'][0][0], places=5)

    def test_display_values(self):
        for obj, shape in [(Vector([1, 2]), (2,)), (Layer([1, 2]), (2,)), (Matrix([[1, 2]]), (1, 2))]:
            values = display_values(obj)
            self.assertEqual(values.shape, shape)
            values.flat[0] = 99
            self.assertEqual(display_values(obj).flat[0], 1)
        # Simulate an older class identity retained across a module reload.
        old_matrix = type('Matrix', (), {'__module__': 'tensor', 'to_list': lambda self: [[1., 2.]]})()
        self.assertEqual(display_values(old_matrix).shape, (1, 2))
        with self.assertRaises(TypeError):
            display_values([1, 2])
        with self.assertRaises(ValueError):
            display_values(Vector([float('nan')]))

    def test_divergence_message(self):
        at = AppTest.from_file(str(ROOT / 'iris_app.py'), default_timeout=120).run()
        with patch.object(Model, 'backward', return_value=float('nan')):
            at.button[0].click().run()
        self.assertFalse(at.exception)
        self.assertIn('diverged', at.error[0].value)

if __name__ == '__main__':
    unittest.main()

