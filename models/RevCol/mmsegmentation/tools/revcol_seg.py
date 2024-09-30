from mmseg.models import build_segmentor

def get_model(cfg):
    model = build_segmentor(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))
    return model