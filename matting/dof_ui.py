import wx
import dof_psd_ui
from pathlib import Path
import skimage
import fnmatch
import os
import psd_matting


class w_frame(dof_psd_ui.DofPsdUI):
    def __init__(self, parent):
        super(w_frame, self).__init__(parent)
        self.m_button5.Bind(wx.EVT_BUTTON, self.on_button_clicked)

    def on_button_clicked(self, event):
        img_lib_path = self.m_textCtrl1.GetValue()
        libpath = Path(img_lib_path)
        is_save_huidu = self.m_checkBox2.IsChecked()
        if not libpath.is_dir():
            print("输入的路径不是一个文件夹")
        elif not libpath.exists():
            print("文件夹不存在")
        psd_list = self.getAllPsd(img_lib_path,'*.psd')
        print('psd数量：',len(psd_list))
        for psd in psd_list:
            psd_path = f"{img_lib_path}/{psd}"
            print(psd_path)
            dir_name =  psd.replace('.psd','')
            outPath = img_lib_path + f'/{dir_name}/'
            mat = psd_matting.AutoMattingPSD(psd= psd_path,outpath= outPath,is_save_huidu=is_save_huidu,psd_name = psd)
            mat.play()
    def getAllPsd(self,path,key):
        res = []
        # for folderName, subFolders, fileNames in os.walk(path):
        #     for filename in fileNames:
        #         if fnmatch.fnmatch(filename,key):
        #             res.append(filename)
        allfile =  os.listdir(path)
        for i in allfile:
            print(i)
        for filename in allfile:
            if fnmatch.fnmatch(filename, key):
                res.append(filename)
        return res

app = wx.App()
window = w_frame(parent=None)
window.Show()
app.MainLoop()