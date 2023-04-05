import wx
import w_frame_xrc
import matting
import skimage
from pathlib import Path


# class MainFrame(wx.Frame):
#     def __int__(self, parent, id, title, size):
#         wx.Frame.__init__(self, parent, id, title, size)
#
#
# class MaApp(wx.App):
#     def __int__(self):
#         wx.App.__init__(self)
#
#     def OnInit(self):
#         frame = MainFrame(None, -1, "My UI", (400, 400))
#         frame.Show()
#         frame.Center(True)
#         return True

# app = wx.App()
# window = w_frame_xrc.DofAutoMatting(parent=None)
# window.Show()
# app.MainLoop()


class w_frame(w_frame_xrc.DofAutoMatting):
    def __init__(self, parent):
        super(w_frame, self).__init__(parent)
        self.m_textCtrl1.Clear()
        self.m_textCtrl11.Clear()
        self.m_button5.Bind(wx.EVT_BUTTON, self.on_button_clicked)

    def on_button_clicked(self, event):
        png1 = self.m_textCtrl1.GetValue()
        png2 = self.m_textCtrl11.GetValue()
        png1path = Path(png1)
        png2path = Path(png2)
        if not png1 and png2:
            print("你没有输入任何路径")
        if png1path.is_dir() or png2path.is_dir():
            print("输入的路径是文件夹")
        elif not (png1path.exists() and png2path.exists()):
            print("存在文件不存在的情况")
        else:
            mat = matting.AutoMatting(png1=png1, png2=png2)
            mat.play()


app = wx.App()
window = w_frame(parent=None)
window.Show()
app.MainLoop()

# mat = matting.AutoMatting(png1="111.png", png2="222.png")
# mat.play()
