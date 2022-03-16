import streamlit as st
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '..')
from models.predict_model import predict


oh_baby = '''Oh baby, baby
Oh baby, baby Oh baby, baby
How was I supposed to know
That something wasn't right here
Oh baby baby
I shouldn't have let you go
And now you're out of sight, yeah
Show me, how you want it to be
Tell me baby
'Cause I need to know now what we've got

My loneliness is killing me
I must confess, I still believe
When I'm not with you I lose my mind
Give me a sign
Hit me baby one more time

Oh baby, baby
The reason I breathe is you
Boy you got me blinded
Oh baby, baby
There's nothing that I wouldn't do
That's not the way I planned it
Show me, how you want it to be
Tell me baby
'Cause I need to know now what we've got

My loneliness is killing me
I must confess, I still believe
When I'm not with you I lose my mind
Give me a sign
Hit me baby one more time

Oh baby, baby
Oh baby, baby
Ah, yeah, yeah
Oh baby, baby
How was I supposed to know
Oh pretty baby
I shouldn't have let you go
I must confess, that my loneliness
Is killing me now
Don't you know I still believe
That you will be here
And give me a sign
Hit me baby one more time

My loneliness is killing me
I must confess, I still believe
When I'm not with you I lose my mind
Give me a sign
Hit me baby one more time

I must confess that my loneliness
Is killing me now
Don't you know I still believe
That you will be here
And give me a sign
Hit me baby one more time
     '''

st.title('Lyrics to Music Genre Classificator')

data_load_state = st.text('Input lyrics of english song')
txt = st.text_area('', oh_baby, height = 380)

classify_button = st.button('Classify')

if classify_button:
    if not txt:
        st.text('You have to input lyrics!!!')
    else:
        pred = predict(txt)
        st.text(f"Song's genre:   {pred}")