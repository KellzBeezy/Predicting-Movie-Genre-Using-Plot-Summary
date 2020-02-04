from tkinter import *
import pred

root = Tk()
# root = Frame(master=root1, bg="blue")
root.configure(background='gray40')

root.title("App Beezy")

#root.pack_propagate(1)
# name = StringVar()

# create the widgets
header = Label(root, text="APP BEEZY", fg="gold", font="Courier 44 bold")
#header.config()

go_btn = Button(root, text="Predict", fg="blue", font="arial 14 bold", width=20, command=lambda: mint1())

title_label = Label(root, text="Title", fg="red")
title_label.config(font=("Arial", 20))
title_entry = Text(root, width=90, font='14', height=2)

plot_label = Label(root, text="Plot", fg="red")
plot_label.config(font=("Arial", 20))
plot_entry = Text(root, width=100, height=10)

keyword_label = Label(root, text="Keywords", fg="red")
keyword_label.config(font=("Arial", 20))
keyword_entry = Text(root, width=100, height=5)

genre_label = Label(root, text="Genre", fg="maroon")
genre_label.config(font=("Arial", 20))
genre_entry = Text(root, width=100, height=2)


# place the widgets
header.grid(row=1, column=30, pady=(10, 5))

title_label.grid(row=4, column=15, sticky=E,  padx=(10, 5))

title_entry.grid(row=4, column=30, pady=(12, 2), padx=(10, 5))

plot_entry.grid(row=7, column=30, pady=(12, 2), padx=(10, 5))

plot_label.grid(row=7, column=15, sticky=E, pady=(10, 5), padx=(10, 5))

keyword_entry.grid(row=10, column=30, pady=(12, 2), padx=(10, 5))

keyword_label.grid(row=10, column=15, sticky=E, pady=(10, 5), padx=(10, 5))

go_btn.grid(row=12, column=30, sticky=E, pady=(10, 5), padx=(5, 5))

genre_label.grid(row=15, column=15, sticky=E, pady=(15, 10), padx=(10, 5))

genre_entry.grid(row=15, column=30, pady=(15, 10), padx=(10, 5))

# def mint():

    #name = ent2.get("1.0", END)
    #title_entry.delete("1.0", END)
    #genre_entry.delete("1.0", END)
    #keyword_entry.delete("1.0", END)
    #keyword_entry.insert(0.0, test.word_tokenize(test.clean_text(test.remove_stopwords(plot_entry.get("1.0", END)))))
    #title_entry.insert(0.0, "COULDN'T FIND THE TITLE !!")
    #genre_entry.insert(0.0, test.infer_tags(plot_entry.get("1.0", END)))

    #print('gui')
    # return name

def mint1():

    #name = ent2.get("1.0", END)
    title_entry.delete("1.0", END)
    genre_entry.delete("1.0", END)
    keyword_entry.delete("1.0", END)
    keyword_entry.insert(0.0, pred.word_tokenize(pred.c_plot(pred.stop_words_fn(plot_entry.get("1.0", END)))))
    title_entry.insert(0.0, "NO TITLE PROVIDED !!")
    if len(plot_entry.get("1.0", END)) < 100:
        print(len(plot_entry.get("1.0", END)))
        genre_entry.insert(0.0, pred.predict1(plot_entry.get("1.0", END)))
    else:
        print(len(plot_entry.get("1.0", END)))
        genre_entry.insert(0.0, pred.predict(plot_entry.get("1.0", END)))

    print('gui')
    # return name


# root.grid()
root.mainloop()
