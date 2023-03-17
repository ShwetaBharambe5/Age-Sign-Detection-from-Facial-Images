# Age-Sign-Detection-from-Facial-Images

Instructions and dependencies to run web development code:

1. Install Xampp server
2. open Xampp control panel and run apache and mysql
3. paste the entire folder in the path C:/xampp/htdocs
4. open browser and run localhost8080:/foldername/index.html
5. you will be redirected to image upload page if you click on upload image
6. you can upload the image by clicking on select option and then click on upload button. then it will notify like uploaded successfully.
7. you will be redirected to output page if you click on output button
8. in the output the image will be displayed.


We tried to integrate php and ml using php-ml library but it did not support all the ml models and we didnt have much time to learn and implement it in django or flask.
So, we could not integrate ml and web dev parts. 

Module Requirements :

1. Opencv
2. Tensorflow 2.x
3. keras 2.2.0
4. dlib
5. shutil ,os ,glob
6. pandas, numpy
7. gc, datetime
8. sklearn


Instructions for the predictions:

1. IMport the prediction_model.py
2. call the predictions() method with model_name param ("inception" or "effb0")
3. This methods returns rectangles (Coordinates) and an info variable (default a string "Red,Blue" describing to use red colorfor acne and Blue color box wrinkles)

4. Format in which the coordinates are:

   Returns : list_of_rectangles_to_be_drawn 
            example: {
              "File_name_1":(
                            {'example_key1_for_acne': [x1,x2,y1,y2] ,... }, ##Draw {{Blue}} Rectangles
                            {'example_key1_for_wrinkle': [x1,x2,y1,y2] ,... } ##Draw {{Red}} Rectangles
              ), 

               "File_name_2":(
                            {'example_key1_for_acne': [x1,x2,y1,y2] ,... }, ##Draw {{Blue}} Rectangles
                            {'example_key1_for_wrinkle': [x1,x2,y1,y2] ,... } ##Draw {{Red}} Rectangles
                            ) 
            }
        ***Note**: These coordinates are regions
        of image (uploaded image) not the image itself 
