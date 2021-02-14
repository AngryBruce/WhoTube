# WhoTube -> Predict Youtubers based on Youtube Thumbnails
Web based Flask Aplication deployed -> https://www.angrybruce.com/

1) Youtube Thumbnails Extractor : Download Thumbnail based on YT channel IDs
2) CNN Training for Colab : Train Xception with ImageNet weights + Image Generators. Apply GradCam to understand the CNN "focus".
3) Flask Webapp to serve the Model
4) Out Scope : Deployment in VM with nginx reverse proxy + gunicorn
