
from mmdet.models.detectors.retinanet import RetinaNet
from mmdet.models.registry import DETECTORS

@DETECTORS.register_module
class Classification(RetinaNet):

    def extract_feat(self, img):
        x, sub_obj = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x, sub_obj

    def forward(self, image, label=None, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(image, label, **kwargs)
        else:
            return self.simple_test(image, label, **kwargs)

    def forward_train(self, img, label, **kwargs):

        x, sub_obj = self.extract_feat(img)
        outs = self.bbox_head(x)
        losses = self.bbox_head.loss(outs, label)
        # print('losses:', losses)
        # assert False,'losses'
        return losses, sub_obj

    def simple_test(self, img, label, **kwargs):
        x, _  = self.extract_feat(img)
        outs = self.bbox_head(x)

        return outs
