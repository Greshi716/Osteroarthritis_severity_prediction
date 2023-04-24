import numpy as np
from keras.models import load_model
from keras.preprocessing import image
class osteroarthritis:
    def __init__(self,filename):
        self.filename=filename

    def predictionosteroarthritis(self):
        model=load_model('modelf.h5')
        imagename=self.filename
        test_image=image.load_img(imagename,target_size=(224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)
        print("result is",result)

        a=max(result)
        b=max(a)
        print("b is",b)
        ans=np.where(a==b)[0][0]
        if ans==1:
            prediction='level1'
            return [{"image": prediction}]
        elif ans ==2:
            prediction = 'level2'
            return [{"image": prediction}]
        elif ans==3:
            prediction='level3'
            return [{"image": prediction}]
        elif ans==4:
            prediction='level4'
            return [{"image": prediction}]
        else:
            prediction='level0'
            return [{"image": prediction}]