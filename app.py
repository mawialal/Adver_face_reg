import streamlit as st 
from facenet_pytorch import MTCNN, InceptionResnetV1
from fake import create_targeted_fake
from vgg import VGG_16
from attack import create_fake

st.title("Upload Image for Adveserial Attack")
uploaded_attack_file = st.file_uploader("Choose an image...", type="jpg",key=1)
if uploaded_attack_file is not None:
    model = VGG_16()
    model.load_weights('vgg_face_torch/VGG_FACE.t7')
    model.eval()
    img,label_orig,label_name = create_fake(model, uploaded_attack_file)
    st.image(img, caption="Aveserial Image : Original "+label_orig+" New Lable : "+label_name)


st.title("Upload your image + image of target")
uploaded_file = st.file_uploader("Choose an image...", type="jpg",key=2)
uploaded_target_file = st.file_uploader("Choose an target image...", type="jpg",key=3)
if uploaded_file is not None and uploaded_target_file is not None :
    model = InceptionResnetV1(pretrained='vggface2').eval().to('cuda')
    
    model.classify = True
    #st.write(uploaded_file)
    a = create_targeted_fake(model, uploaded_file, uploaded_target_file)
    st.image(a, caption="Aveserial Image")
 
