from keras import Model
from keras.applications import VGG16, VGG19, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, \
    DenseNet169, DenseNet201, NASNetMobile, NASNetLarge, ResNet50, Xception

from pyspec.machine.model.cnn import SingleInputCNNModel


class Resnet50CNNModel(SingleInputCNNModel):

    def build(self) -> Model:
        model = ResNet50(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class XceptionModel(SingleInputCNNModel):
    """
    keras XCEPTION model
    """

    def build(self) -> Model:
        model = Xception(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class VGG16Model(SingleInputCNNModel):

    def build(self) -> Model:
        model = VGG16(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class VGG19Model(SingleInputCNNModel):

    def build(self) -> Model:
        model = VGG19(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class InceptionModel(SingleInputCNNModel):

    def build(self) -> Model:
        model = InceptionV3(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class InceptionResNetModel(SingleInputCNNModel):

    def build(self) -> Model:
        model = InceptionResNetV2(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class MobileNetModel(SingleInputCNNModel):

    def build(self) -> Model:
        model = MobileNet(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class MobileNetV2Model(SingleInputCNNModel):

    def build(self) -> Model:
        model = MobileNetV2(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class DenseNet121Model(SingleInputCNNModel):

    def build(self) -> Model:
        model = DenseNet121(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class DenseNet169Model(SingleInputCNNModel):

    def build(self) -> Model:
        model = DenseNet169(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class DenseNet201Model(SingleInputCNNModel):

    def build(self) -> Model:
        model = DenseNet201(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class NASNetMobileModel(SingleInputCNNModel):

    def build(self) -> Model:
        model = NASNetMobile(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class NASNetLargeModel(SingleInputCNNModel):

    def build(self) -> Model:
        model = NASNetLarge(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model
