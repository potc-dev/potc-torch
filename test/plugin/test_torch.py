import pytest
import torch
from potc.testing import provement
from potc.translate import BlankTranslator

from potc_torch.plugin import __rules__


@pytest.mark.unittest
class TestPlugin(provement(BlankTranslator(__rules__))):
    def test_torch_dtype(self):
        with self.transobj_assert(torch.float) as (obj, name):
            assert obj == torch.float
            assert name == 'torch_dtype'

        with self.transobj_assert(torch.float16) as (obj, name):
            assert obj == torch.float16
            assert name == 'torch_dtype'

        with self.transobj_assert(torch.float32) as (obj, name):
            assert obj == torch.float32
            assert name == 'torch_dtype'

        with self.transobj_assert(torch.float64) as (obj, name):
            assert obj == torch.float64
            assert name == 'torch_dtype'

        with self.transobj_assert(torch.int) as (obj, name):
            assert obj == torch.int
            assert name == 'torch_dtype'

        with self.transobj_assert(torch.int8) as (obj, name):
            assert obj == torch.int8
            assert name == 'torch_dtype'

        with self.transobj_assert(torch.int16) as (obj, name):
            assert obj == torch.int16
            assert name == 'torch_dtype'

        with self.transobj_assert(torch.int32) as (obj, name):
            assert obj == torch.int32
            assert name == 'torch_dtype'

        with self.transobj_assert(torch.int64) as (obj, name):
            assert obj == torch.int64
            assert name == 'torch_dtype'

    def test_torch_size(self):
        with self.transobj_assert(torch.Size([])) as (obj, name):
            assert obj == torch.Size([])
            assert name == 'torch_size'

        with self.transobj_assert(torch.Size([2, 3, 5, 7])) as (obj, name):
            assert obj == torch.Size([2, 3, 5, 7])
            assert name == 'torch_size'

    def test_torch_tensor(self):
        int_tensor = torch.randint(-100, 300, (3, 4, 2), dtype=torch.float16)
        with self.transobj_assert(int_tensor) as (obj, name):
            assert obj.dtype == int_tensor.dtype
            assert (obj == int_tensor).all()
            assert name == 'torch_tensor'

        float_tensor = torch.randn((3, 5, 4), dtype=torch.float64)
        with self.transobj_assert(float_tensor) as (obj, name):
            assert obj.dtype == float_tensor.dtype
            assert torch.isclose(obj, float_tensor, rtol=1e-4, atol=1e-4).all()
            assert name == 'torch_tensor'
