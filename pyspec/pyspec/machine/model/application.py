from keras import Model
from keras.applications import VGG16, VGG19, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, \
    DenseNet169, DenseNet201, NASNetMobile, NASNetLarge, ResNet50, Xception

from pyspec.machine.model.cnn import CNNClassificationModel


class Resnet50CNNModel(CNNClassificationModel):

    def build(self) -> Model:
        model = ResNet50(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class XceptionModel(CNNClassificationModel):
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


class VGG16Model(CNNClassificationModel):

    def build(self) -> Model:
        model = VGG16(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class VGG19Model(CNNClassificationModel):

    def build(self) -> Model:
        model = VGG19(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class InceptionModel(CNNClassificationModel):

    def build(self) -> Model:
        model = InceptionV3(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class InceptionResNetModel(CNNClassificationModel):

    def build(self) -> Model:
        model = InceptionResNetV2(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class MobileNetModel(CNNClassificationModel):

    def build(self) -> Model:
        model = MobileNet(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class MobileNetV2Model(CNNClassificationModel):

    def build(self) -> Model:
        model = MobileNetV2(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class DenseNet121Model(CNNClassificationModel):

    def build(self) -> Model:
        model = DenseNet121(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class DenseNet169Model(CNNClassificationModel):

    def build(self) -> Model:
        model = DenseNet169(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class DenseNet201Model(CNNClassificationModel):

    def build(self) -> Model:
        model = DenseNet201(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class NASNetMobileModel(CNNClassificationModel):

    def build(self) -> Model:
        model = NASNetMobile(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model


class NASNetLargeModel(CNNClassificationModel):

    def build(self) -> Model:
        model = NASNetLarge(
            include_top=True,
            weights=None,
            input_shape=(self.width, self.height, self.channels),
            classes=2
        )

        return model
