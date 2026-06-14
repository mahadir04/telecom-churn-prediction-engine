"""
shap_patch.py
-------------
Monkeypatch SHAP to support newer XGBoost versions where learner_model_param["base_score"]
is represented as a bracketed string (e.g. "[5E-1]").
"""
try:
    import shap.explainers._tree
    original_decode = shap.explainers._tree.decode_ubjson_buffer

    def patched_decode(*args, **kwargs):
        res = original_decode(*args, **kwargs)
        try:
            if "learner" in res and "learner_model_param" in res["learner"]:
                param = res["learner"]["learner_model_param"]
                if "base_score" in param:
                    bs = param["base_score"]
                    if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
                        param["base_score"] = bs[1:-1]
        except Exception:
            pass
        return res

    shap.explainers._tree.decode_ubjson_buffer = patched_decode
except ImportError:
    pass
