
def FlyingCameraNetworkDisplay():


    import random

    import pandas as pd
    import pickle as pkl
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
    from random import randrange
    import random
    from ipywidgets import interactive, interact , FloatSlider,widgets,VBox,HBox,interactive_output,Layout,Output,GridBox
    from IPython.display import display,HTML,clear_output,Markdown
    from scipy.interpolate import UnivariateSpline
    
    clear_output()
    print("loading...(Might take a minute)")

    #Network layers
    def hyperspectral(input_,n_o_input, keep_prob, filter_width = 1, stride_size =1, relu_alpha = 0.2):
        n_o_strides = int((n_o_input-filter_width)/stride_size) +1  #round down

        Hyper_layer = []

        def dense(input_,start,keep_prob, filter_width, stride_size, relu_alpha):
            nn_input = tf.slice(input_,[0,start],[-1,filter_width])

            dropout1 = tf.nn.dropout(nn_input, keep_prob)
            dense1 = tf.layers.dense(dropout1, 1)
            relu1 = tf.maximum(relu_alpha * dense1, dense1)        
            return relu1

        for step in range(n_o_strides):
            start = step*stride_size
            output = dense(input_,start,keep_prob, filter_width, stride_size, relu_alpha)
            Hyper_layer.append(output)

        if (n_o_input-filter_width)%stride_size>0:
            start = n_o_input-filter_width
            output = dense(input_,start,keep_prob, filter_width, stride_size, relu_alpha)
            Hyper_layer.append(output)

        Hyper_l_stacked = tf.concat(Hyper_layer,1)

        return Hyper_l_stacked , n_o_strides
    def Classifier(input_,n_o_class,n_o_input, keep_prob,relu_alpha = 0.2):
        if n_o_input == 3:
            is_RGB = True
        elif n_o_input == 571:
            is_RGB = False
        else:
            raise ValueError('A very specific bad thing happened.'+str(n_o_input))

        if is_RGB:
            dense0 = tf.layers.dense(input_, 3)    
            relu0 = tf.maximum(relu_alpha * dense0, dense0)
            first_layer_out = tf.nn.dropout(relu0, keep_prob)
        else:
            first_layer_out,n_o_input= hyperspectral(input_,n_o_input, keep_prob, filter_width = 30, stride_size =1, relu_alpha = 0.2)

        hidden_size = n_o_input*2/3
        hidden_nodes = int(hidden_size)+1 # rounding


        dense1 = tf.layers.dense(first_layer_out, hidden_nodes)    
        relu1 = tf.maximum(relu_alpha * dense1, dense1)
        dropout1 = tf.nn.dropout(relu1, keep_prob)


        class_logits = tf.layers.dense(dropout1, n_o_class)    

        return class_logits
    def softmax(input_,m_class, n_class,n_o_input, keep_prob,relu_alpha = 0.2,sub_scaling = 1):

        n_o_class = m_class+n_class

        #raw output
        logits= Classifier(input_,n_o_class,n_o_input, keep_prob,relu_alpha = 0.2)
        subclass_softmax = tf.nn.softmax(logits)

        #Reduce outputs from 8 subclasses to 2 main classes
        m_class_logit, n_class_logit = tf.split(logits, [m_class, n_class], 1)
        m_class_logit1 =tf.reduce_sum(m_class_logit,1, keepdims =True) 
        n_class_logit1 =tf.reduce_sum(n_class_logit,1, keepdims =True) 
        main_class_logits = tf.concat([m_class_logit1, n_class_logit1], 1)
        main_class_softmax = tf.nn.softmax(main_class_logits)

        return subclass_softmax,main_class_softmax




    #Defining the training parameters
    tf.reset_default_graph()

    n_class = 3
    m_class = 5
    n_o_class = m_class+n_class
    epochs = 10
    keep_probability = 0.95
    sub_scaling = 1 
    g1 = tf.Graph()
    with g1.as_default() as g_1:
        H_keep_prob = tf.placeholder(tf.float32)

        H_n_o_input = 571
        H_input_ = tf.placeholder(tf.float32,  [None,H_n_o_input])
        H_subclass_softmax,H_main_class_softmax =softmax(H_input_,m_class, n_class,H_n_o_input, H_keep_prob,relu_alpha = 0.2,sub_scaling = sub_scaling) 
    #Relectance spectrum
    g2 = tf.Graph()
    with g2.as_default() as g_2:
        R_keep_prob = tf.placeholder(tf.float32)
        R_n_o_input = 3
        R_input_ = tf.placeholder(tf.float32,  [None,R_n_o_input])
        R_subclass_softmax,R_main_class_softmax =softmax(R_input_,m_class, n_class,R_n_o_input, R_keep_prob,relu_alpha = 0.2,sub_scaling = sub_scaling) 




    #Display network result
    clear_output()

    

    display(Markdown("## &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;View Classification" ))

    RGB = []
    hyperspectral = []

    box1 = Output(layout=Layout(width='100%',border ='1px solid #E0E0E0',padding = '0px 0px 0px 20px',fill = "black"))
    box2 = Output(layout=Layout(width='100%',border ='1px solid #E0E0E0',padding = '0px 20px 0px 20px'))
    box3 = Output(layout=Layout(width='100%',border ='1px solid #E0E0E0',padding = '0px 0px 0px 20px'))
    box4 = Output(layout=Layout(width='100%',border ='1px solid #E0E0E0',padding = '0px 0px 0px 20px'))
    box5 = Output(layout=Layout(width='100%',border ='1px solid #E0E0E0',padding = '0px 20px 0px 20px'))
    box6 = Output(layout=Layout(width='100%',border ='1px solid #E0E0E0',padding = '0px 20px 0px 20px'))
    
    markdown =  Output(layout=Layout(width='95%'))
    with markdown:
        display(Markdown("#### Adjust The Wavelengths"))
        display(Markdown("<sup>To Customise Your Own RGB & Hyperspectral Input</sup>"))
    
    slider1 = FloatSlider(min=0, max=1, step=0.01, value=0.5,description='380nm',layout= Layout(width='99%'))
    slider2 = FloatSlider(min=0, max=1, step=0.01, value=0.5,description='450nm',layout= Layout(width='99%'))
    slider3 = FloatSlider(min=0, max=1, step=0.01, value=0.5,description='550nm',layout= Layout(width='99%'))
    slider4 = FloatSlider(min=0, max=1, step=0.01, value=0.5,description='650nm',layout= Layout(width='99%'))
    slider5 = FloatSlider(min=0, max=1, step=0.01, value=0.5,description='750nm',layout= Layout(width='99%'))
    slider6 = FloatSlider(min=0, max=1, step=0.01, value=0.5,description='850nm',layout= Layout(width='99%'))
    slider7 = FloatSlider(min=0, max=1, step=0.01, value=0.5,description='950nm',layout= Layout(width='99%'))

    def click(event):
        global hyperspectral
        global RGB
        classify(RGB,hyperspectral)

    button = widgets.Button(
        description='Run Classification',
        layout={'width': '130px','margin':'0px 0px 10px 15%'}
    )
    button.on_click(click)
    slider_list = [markdown,slider1,slider2,slider3,slider4,slider5,slider6,slider7,button]

    for slider in slider_list:
        with box3:
            display(slider)

    def classify(RGB,hyperspectral):
        keep_probability = 1

        def print_network_output(save_model_path,title,subclass_softmax,main_class_softmax,input_,reflectance_data,n_o_input,g,keep_prob):
            clear_output(wait=True)
            print("""Loading...(Might take a minute)
*The next classification doesn't load until this one is completed""")

            with tf.Session(graph=g) as sess:
                sess.run(tf.global_variables_initializer())

                loader = tf.train.import_meta_graph(save_model_path + '.meta')
                loader.restore(sess, save_model_path)
                subclass_softmax_p,main_class_softmax_p= sess.run([subclass_softmax,main_class_softmax], feed_dict = {input_:reflectance_data,keep_prob:keep_probability})
                clear_output(wait=True)
                display(Markdown("#### "+title))
                display(Markdown("##### Main Class"))
                main_class_softmax_p  = main_class_softmax_p[0]
                display(Markdown("Non-scat: "+str(main_class_softmax_p[0])))
                display(Markdown("Scat    : "+str(main_class_softmax_p[1])))
                display(Markdown("##### Subclass"))
                subclass_softmax_p = subclass_softmax_p[0]

                subclass_name = [    "Leaf Litter   ",
                                     "Ground        ",
                                     "Wood          ",
                                     "Bird Scat     ",
                                     "Mammal Scat   ",
                                     "Amphibian Scat",
                                     "Reptile Scat  ",
                                     "Reptile Urea  "] 
                for subclass,name in zip(subclass_softmax_p,subclass_name):
                    display(Markdown(name+": " +str(subclass)))
        with box4:
            print_network_output('../training_rgb',
                                 "RGB Network Classification",
                                 R_subclass_softmax,
                                 R_main_class_softmax,
                                 R_input_,[RGB],
                                 3,
                                 g_2,
                                 R_keep_prob)
        with box5:
            print_network_output('../training_Hyperspectral',
                                 "Hyperspectral Network Classification",
                                 H_subclass_softmax,
                                 H_main_class_softmax,
                                 H_input_,
                                 [hyperspectral],
                                 571,
                                 g_1,
                                 H_keep_prob)           
    def f(z,a,b,c,d,e,f,g):
        if z == "Select":
            for slider in slider_list:
                slider.layout.display = 'none'
            return 0
        data_path ="../Training data/averaged all.csv"
        data = pd.read_csv(data_path)
        data.head()
        target_fields ='Class'
        data = data.drop(["Class code"],axis=1)
        features0, targets0 = data.drop(target_fields, axis=1), data[target_fields]
        features, targets  = np.array(features0) , np.array(targets0)

        dic = {}
        for feature,target in zip(features, targets):
            dic[target]=list(feature)
        if z == "Make your own":
            x= [380,450,550,650,750,850,950]
            y = [a,b,c,d,e,f,g]
            plt.scatter(x, y)
            plt.plot(x, y, 'o')
        else:
            x= [x for x in range(380,951)]
            y = dic[z.replace("Average ","")]     
            for slider in slider_list:
                slider.layout.display = 'none'

        graph = UnivariateSpline(x, y, s=0)
        xs = np.linspace(380, 950, 100)
        ys = graph(xs)

        with box2:
            clear_output(wait=True)
            display(Markdown("#### Hyperspectral Reflectance" ))
            plt.ylim(0,1)
            plt.xlim(380, 950)
            plt.plot(xs, ys)
            plt.title("Reflectance Intensity")
            plt.show()

        global hyperspectral
        global RGB
        hyperspectral = [ min(max(round(float(graph(x)),7),0),1) for x in range(380,951)]
        red =  min(max(round(float(graph(680)),4),0),1)
        green =  min(max(round(float(graph(530)),4),0),1)
        blue =  min(max(round(float(graph(465)),4),0),1)
        RGB = [red,green,blue]

        with box1:
            clear_output(wait=True)
            red_rgb = int(red*255)
            green_rgb = int(green*255)
            blue_rgb = int(blue*255)
            display(Markdown("#### RGB Reflectance" ))
            html = '<svg height="100" width="100%"><rect x="0" y="10px" width="100" height="100" fill="rgb('+ str(red_rgb)+","+str(green_rgb)+","+str(blue_rgb)+')" /></svg>'     
            display(HTML(html))
            display(Markdown("R(680nm): "+ str(red) ))
            display(Markdown("G(530nm): "+ str(green) ))
            display(Markdown("B(465nm): "+ str(blue) ))

        if z == "Make your own":
            with box4:
                clear_output(wait=False)
            with box5:
                clear_output(wait=False)
            for slider in slider_list:
                slider.layout.display = 'flex'
        else:
            classify(RGB,hyperspectral)
    material = widgets.Dropdown(
        options=["Select",
            "Make your own",
              'Average Amphibian Scat', 
              'Average Bird Scat', 
              'Average Ground',
              "Average Leaf Litter",
              "Average Mammal Scat",
              "Average Reptile Scat",
              "Average Reptile Urea",
              "Average Wood"],
        value='Select',
        description='Material:',
    )
    output = interactive_output(f, {"z":material,'a':slider1,'b' :slider2, 'c':slider3,'d':slider4,'e' :slider5, 'f':slider6,'g':slider7})
    display(material)
    grid = GridBox(children=[box1,box2,box3,box4,box5,box6],
            layout=Layout(
                width='97%',
                grid_template_columns='30% 30% 30%',
                grid_template_rows='auto auto',
                grid_gap='5px 15px',
                margin = "10px 0px 0px 25px"
            )
           )
    display(grid)