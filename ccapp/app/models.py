from django.db import models


class UploadFile(models.Model):
    """Model for Uoloaded File"""
    file = models.ImageField('ImageFile')

    def __str__(self):
        """Return URL of File"""
        return self.file.url
