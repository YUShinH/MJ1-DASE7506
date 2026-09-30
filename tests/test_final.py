import torch
from test_contract import ContractTests as BaseContract
from student import build_model


class FinalTests(BaseContract):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.config=dict(vocab=2048,width=32,heads=4,depth=2,context=256,
                         backbone='relu2',position_encoding='rope',dropout=0.,variant='gated_cache')
        self.model=build_model(self.config).eval()

    def test_prefix_matches_full_window(self):
        ids=torch.randint(0,2048,(2,256))
        with torch.no_grad():
            full=self.model.predict_log_probs(ids)
            short=self.model.predict_log_probs(ids[:1,:17])
        torch.testing.assert_close(full[:1,:17],short,atol=1e-5,rtol=1e-5)

    def test_gate_can_train_without_changing_backbone(self):
        ids=torch.randint(0,2048,(2,32))
        for name,p in self.model.named_parameters():
            p.requires_grad_(name.startswith('gate.'))
        before={n:p.detach().clone() for n,p in self.model.named_parameters() if not n.startswith('gate.')}
        optimizer=torch.optim.AdamW(self.model.gate.parameters(),lr=.001)
        loss=-self.model.predict_log_probs(ids).gather(-1,ids.unsqueeze(-1)).mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.model.gate.parameters()))
        optimizer.step()
        for n,p in self.model.named_parameters():
            if n in before:
                torch.testing.assert_close(p,before[n],atol=0,rtol=0)


del BaseContract
