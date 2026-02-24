import customtkinter as ctk





class PasswordGeneratorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("密码生成器")
        self.geometry("400x350")
        self.setup_ui()

    def setup_ui(self):
        self.label = ctk.CTkLabel(self, text="密码生成器")
        self.label.pack(pady=20)





if __name__ == "__main__":
    app = PasswordGeneratorApp()
    app.mainloop()




