
from mmdet.models.detectors.retinanet import RetinaNet
from mmdet.models.registry import DETECTORS

@DETECTORS.register_module
class NASRetinaNetTrain(RetinaNet):

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, image, depth=None, focal=None, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(image, depth, focal, **kwargs)
        else:
            return self.simple_test(image, depth, focal, **kwargs)

    def forward_train(self, img, depth, focal, **kwargs):

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        losses = self.bbox_head.loss(outs, depth)
        # print('losses:', losses)
        # assert False,'losses'
        return losses

    def simple_test(self, img, depth, focal, **kwargs):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        return outs
