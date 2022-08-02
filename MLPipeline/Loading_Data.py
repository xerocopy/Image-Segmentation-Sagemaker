import matplotlib.pyplot as plt
import io


class Loading_Data:

    def loading_data(self, s3, default_location):
        img_data_array = []
        mask_data_stack = []
        print("Reading the images")
        s3_bucket = "appledatabucket"
        keys = []
        for obj in s3.Bucket(s3_bucket).objects.all():
            keys.append(obj.key)
        for key in keys:
            file_stream = io.BytesIO()
            s3.Bucket(s3_bucket).Object(key).download_fileobj(file_stream)
            if ".jpg" in key and "Apple" in key:
                print(key)
                img = plt.imread(file_stream, format='jpg')
                print(img.shape)
                img_data_array.append(img)

            elif ".tiff" in key and "Apple" in key:
                mask = plt.imread(file_stream, format='tiff')
                print(mask.shape)
                mask_data_stack.append(mask)

        return img_data_array, mask_data_stack
