import PySimpleGUI as sg
import os.path
import joblib

sg.theme('reddit')   # Add a touch of color

# Create two column
column_one = [
    [
        sg.Text("Fill data below"),
    ],
    [
        sg.Text("Age"),
        sg.InputText(key="-AGE-", do_not_clear=True, size=(10,1)),
    ],
    [
        sg.Text("Balance"),
        sg.InputText(key="-BALANCE-", do_not_clear=True, size=(10,1)),
    ],
    [
        sg.Text("Credit Score"),
        sg.InputText(key="-CREDIT-", do_not_clear=True, size=(10,1)),
    ],
    [
        sg.Text("Active Member?"),
        sg.Checkbox('Yes', default=False, key="-YES-"),
        sg.Checkbox('No', default=False, key="-NO-"),
    ],
    [
        sg.Button("PREDICT"), sg.Text("Result : ", key="-RESULT-"),
    ],
    [
        sg.Text("Exit Probability : ", key="-PROBRES-"),
    ],
]

column_two = [
    #[
     #   sg.Button("DATA PREVIEW"),
    #],
    [
        sg.Button("TRAINING"),
    ],
    [
        sg.Button("MODEL PERFORMANCE"),
    ],

]

# ----- Full layout -----
layout = [
    # ROW 1
    [
        sg.Column(column_one),
        sg.VSeperator(),
        sg.Column(column_two),
    ],

    # ROW 2
    [
        sg.Button("EXIT")
    ]
]

window = sg.Window("ANN Kelompok 8", layout)

# Create an event loop
while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "EXIT" or event == sg.WIN_CLOSED:
        break
    
    if event == "TRAINING":
        from subprocess import call
        call(["python3", "training_gui.py"])

    if event == "DATA PREVIEW":
        from subprocess import call
        call(["python3", "data_preview.py"])

    if event == "MODEL PERFORMANCE":
        from subprocess import call
        call(["python3", "performance.py"])

    if event == "PREDICT" :
        if values["-YES-"] == True:
            active = 1
        elif values["-NO-"] == True:
            active = 0
        
        age = values["-AGE-"]
        balance = values["-BALANCE-"]
        credit = values["-CREDIT-"]

        # Load model
        to_predict = [0, 0, 0, 0]

        import numpy as np
        to_predict[0] = age
        to_predict[1] = balance
        to_predict[2] = active
        to_predict[3] = credit

        import pickle as pkl
        with open("scaler.pkl", "rb") as infile:
            scaler = pkl.load(infile)
            to_predict = np.array(to_predict)
            to_predict = to_predict.reshape(1, -1)
            to_predict = scaler.transform(to_predict)

        from tensorflow import keras
        model = keras.models.load_model('./model/ann.h5')

        get_result = model.predict(to_predict)
        final_result = 0
        print(get_result[0][0])
        if(get_result[0][0] >= 0.5):
            final_result = 1
        if(get_result[0][0] < 0.5):
            final_result = 0


        if final_result == 1:
            window['-RESULT-'].update('Result : EXIT', background_color= "red", text_color="white")
        if final_result == 0:
            window['-RESULT-'].update('Result : STAY', background_color= "green", text_color="white")

        probtext = "Exit Probabilit : " + str(get_result[0][0])
        window['-PROBRES-'].update(probtext)

window.close()