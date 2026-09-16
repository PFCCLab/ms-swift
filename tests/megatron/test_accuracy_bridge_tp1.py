"""Instance-scoped TP1 bridge behavior, including mixed configurations."""
import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def load_patch():
    path = Path(__file__).resolve().parents[2] / 'swift/megatron/init.py'
    node = next(
        node for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == '_patch_mcore_bridge_tp1_accuracy')
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[node.name]


class TestBridgeTp1Patch(unittest.TestCase):

    def setUp(self):
        self.modules = []
        for name, class_name in (('mtp_layer', 'MultiTokenPredictionLayer'), ('transformer_block', 'TransformerBlock')):
            module = ModuleType(name)
            module.make_viewless_tensor = lambda inp, **kwargs: ('wrapped', inp)
            module.gather_from_tensor_model_parallel_region = lambda inp, **kwargs: ('gathered', inp)

            def forward(self, inp, callback=None, module=module):
                result = module.make_viewless_tensor(inp=inp, requires_grad=True, keep_graph=True)
                if callback:
                    callback()
                return result

            cls = type(class_name, (), {'forward': forward, '_concat_embeddings': forward, '_get_embeddings': forward})
            setattr(module, class_name, cls)
            self.modules.append((module, cls))
        modules = ModuleType('mcore_bridge.model.modules')
        modules.mtp_layer, modules.transformer_block = [item[0] for item in self.modules]
        replacement = patch.dict(sys.modules, {'mcore_bridge.model.modules': modules})
        replacement.start()
        self.addCleanup(replacement.stop)
        self.apply_patch = load_patch()
        self.apply_patch()

    def instance(self, cls, enabled, tp_size=1):
        instance = cls()
        instance.config = SimpleNamespace(dsa_accuracy_compatible=enabled, tensor_model_parallel_size=tp_size)
        return instance

    def test_idempotent(self):
        originals = [(module.make_viewless_tensor, cls.forward) for module, cls in self.modules]
        self.apply_patch()
        for (module, cls), (function, method) in zip(self.modules, originals):
            self.assertIs(module.make_viewless_tensor, function)
            self.assertIs(cls.forward, method)

    def test_only_explicit_dsa_tp1_instance_skips_viewless(self):
        inp = object()
        for module, cls in self.modules:
            name = '_concat_embeddings' if module.__name__ == 'mtp_layer' else 'forward'
            for enabled, tp_size in ((False, 1), (True, 1), (True, 2)):
                output = getattr(self.instance(cls, enabled, tp_size), name)(inp)
                if enabled and tp_size == 1:
                    self.assertIs(output, inp)
                else:
                    self.assertEqual(output, ('wrapped', inp))
            self.assertEqual(module.make_viewless_tensor(inp, True, True), ('wrapped', inp))

    def test_gather_uses_instance_scope_and_actual_group(self):
        module, cls = self.modules[1]
        inp = object()
        for enabled, tp_size in ((False, 1), (True, 1), (True, 2)):
            observed = []
            group = SimpleNamespace(size=lambda: tp_size)

            def callback():
                observed.append(module.gather_from_tensor_model_parallel_region(inp, group))

            self.instance(cls, enabled).forward(inp, callback=callback)
            if enabled and tp_size == 1:
                self.assertIs(observed[0], inp)
            else:
                self.assertEqual(observed[0], ('gathered', inp))
        self.assertEqual(module.gather_from_tensor_model_parallel_region(inp), ('gathered', inp))

    def test_nested_legacy_call_and_exception_restore_scope(self):
        module, cls = self.modules[1]
        enabled = self.instance(cls, True)
        legacy = self.instance(cls, False)
        inp = object()

        def nested():
            self.assertEqual(legacy.forward(inp), ('wrapped', inp))
            self.assertIs(module.make_viewless_tensor(inp, True, True), inp)
            raise ValueError('test failure')

        with self.assertRaisesRegex(ValueError, 'test failure'):
            enabled.forward(inp, callback=nested)
        self.assertEqual(module.make_viewless_tensor(inp, True, True), ('wrapped', inp))


if __name__ == '__main__':
    unittest.main()
