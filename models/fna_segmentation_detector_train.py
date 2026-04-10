
from mmdet.models.detectors.retinanet import RetinaNet
from mmdet.models.registry import DETECTORS

@DETECTORS.register_module
class NASSegmentationTrain(RetinaNet):

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, image, label=None, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(image, label, **kwargs)
        else:
            return self.simple_test(image, label, **kwargs)

    def forward_train(self, img, lab, **kwargs):
        # print('forward_train img.shape', img.shape)  # [2, 3, 768, 768]
        # print('forward_train lab.shape', lab.shape)  # [2, 768, 768]
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        losses = self.bbox_head.loss(outs, lab)
        # print('losses:', losses)
        # assert False,'losses'
        return losses

    def simple_test(self, img, lab, **kwargs):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        return outs
