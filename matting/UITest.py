import wx
import w_frame_xrc
import matting
import skimage
from pathlib import Path
import fnmatch
import os
import re


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
        # self.m_textCtrl1.Clear()
        # self.m_textCtrl11.Clear()
        self.m_button5.Bind(wx.EVT_BUTTON, self.on_button_clicked)

    def on_button_clicked(self, event):
        # png1 = self.m_textCtrl1.GetValue()
        # png2 = self.m_textCtrl11.GetValue()
        # png1path = Path(png1)
        # png2path = Path(png2)
        # if not png1 and png2:
        #     print("你没有输入任何路径")
        # if png1path.is_dir() or png2path.is_dir():
        #     print("输入的路径是文件夹")
        # elif not (png1path.exists() and png2path.exists()):
        #     print("存在文件不存在的情况")
        # else:
        #     mat = matting.AutoMatting(png1=png1, png2=png2)
        #     mat.play()


        img_lib_path = self.m_textCtrl1.GetValue()
        libpath = Path(img_lib_path)
        if not libpath.is_dir():
            print("输入的路径不是一个文件夹")
        elif not libpath.exists():
            print("文件夹不存在")
        img_dic = {}
        imgAlist = self.getAllPng(img_lib_path,'*_A.png')
        imgBlist = self.getAllPng(img_lib_path,'*_B.png')
        for img in imgAlist:
            allstr = re.split('\.|_', img)
            Bpath = (allstr[0]+'_B.'+allstr[2])
            if len(allstr) !=3:
                print(allstr + "该为文件命名格式不对")
                continue
            if allstr[0] in img_dic.keys():
                continue
            elif not Bpath in imgBlist:
                print(img + "没有B文件")
                continue
            else:
                img_dic[allstr[0]] = (img_lib_path + f"/{img}",img_lib_path + f'/{Bpath}')

        i = 0
        for imgAB in img_dic.values():
            i+=1
            outPath = img_lib_path + f'/{i}/'
            mat = matting.AutoMatting(png1=imgAB[0], png2=imgAB[1],outpath= outPath)
            mat.play()
    def getAllPng(self,path,key):
        res = []
        for folderName, subFolders, fileNames in os.walk(path):
            for filename in fileNames:
                if fnmatch.fnmatch(filename,key ):
                    res.append(filename)
        return res

app = wx.App()
window = w_frame(parent=None)
window.Show()
app.MainLoop()

# mat = matting.AutoMatting(png1="111.png", png2="222.png")
# mat.play()
