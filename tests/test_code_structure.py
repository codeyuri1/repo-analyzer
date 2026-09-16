from repo_agent_chat.code_structure import (
    extract_code_structures,
    extract_python_structures,
)


def test_extract_python_structures_generaliza_para_outro_dominio() -> None:
    evidence = [
        "payments/service.py:1: class PaymentService:",
        "payments/service.py:2:     def process(self, payment):",
        "payments/service.py:3:         receipt = self.gateway.charge(payment)",
        "payments/service.py:4:         return Receipt(receipt)",
    ]

    structures = extract_python_structures(evidence)

    assert len(structures) == 1
    assert structures[0].path == "payments/service.py"
    assert "PaymentService" in structures[0].symbols
    assert "process" in structures[0].symbols
    assert "payment" in structures[0].symbols
    assert "self.gateway.charge" in structures[0].symbols
    assert "Receipt" in structures[0].symbols


def test_extract_python_structures_usa_fallback_para_chunk_incompleto() -> None:
    evidence = [
        "orders.py:40:     def approve(self, order_id):",
        "orders.py:41:         return self.repository.save(order_id)",
    ]

    structures = extract_python_structures(evidence)

    assert "approve" in structures[0].symbols
    assert "self.repository.save" in structures[0].symbols


def test_extract_code_structures_analisa_java_sem_configuracao_de_dominio() -> None:
    evidence = [
        "src/PaymentService.java:1: public class PaymentService {",
        "src/PaymentService.java:2:   Receipt process(Payment payment) {",
        "src/PaymentService.java:3:     return gateway.charge(payment);",
        "src/PaymentService.java:4:   }",
        "src/PaymentService.java:5: }",
    ]

    structures = extract_code_structures(evidence)

    assert structures[0].path == "src/PaymentService.java"
    assert "PaymentService" in structures[0].symbols
    assert "process" in structures[0].symbols
    assert "gateway.charge" in structures[0].symbols


def test_extract_code_structures_ignora_documentacao() -> None:
    structures = extract_code_structures(
        ["README.md:1: class IstoNaoETratadoComoCodigo"]
    )

    assert structures == []
