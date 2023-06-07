# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 3.10.1-0-g8feb16b3)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
## Class DofPsdUI
###########################################################################

class DofPsdUI ( wx.Frame ):

	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Dof psd资源解析", pos = wx.DefaultPosition, size = wx.Size( 500,231 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

		bSizer2 = wx.BoxSizer( wx.VERTICAL )

		sbSizer3 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, wx.EmptyString ), wx.VERTICAL )

		sbSizer2 = wx.StaticBoxSizer( wx.StaticBox( sbSizer3.GetStaticBox(), wx.ID_ANY, wx.EmptyString ), wx.HORIZONTAL )


		sbSizer2.Add( ( 0, 0), 1, wx.EXPAND, 5 )

		self.m_staticText1 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, u"psd资源库路径:", wx.Point( -1,-1 ), wx.DefaultSize, 0 )
		self.m_staticText1.Wrap( -1 )

		sbSizer2.Add( self.m_staticText1, 0, wx.ALL, 5 )

		self.m_textCtrl1 = wx.TextCtrl( sbSizer2.GetStaticBox(), wx.ID_ANY, u"G:\\psd_lib\\tttt", wx.Point( -1,-1 ), wx.Size( 300,-1 ), 0 )
		sbSizer2.Add( self.m_textCtrl1, 0, wx.ALL, 5 )


		sbSizer2.Add( ( 0, 0), 1, wx.EXPAND, 5 )


		sbSizer3.Add( sbSizer2, 1, wx.EXPAND, 5 )

		sbSizer31 = wx.StaticBoxSizer( wx.StaticBox( sbSizer3.GetStaticBox(), wx.ID_ANY, wx.EmptyString ), wx.HORIZONTAL )


		sbSizer31.Add( ( 0, 0), 1, wx.EXPAND, 5 )

		self.m_checkBox2 = wx.CheckBox( sbSizer31.GetStaticBox(), wx.ID_ANY, u"是否生成灰度图和带框原图", wx.DefaultPosition, wx.DefaultSize, 0 )
		sbSizer31.Add( self.m_checkBox2, 0, wx.ALL, 5 )


		sbSizer31.Add( ( 0, 0), 1, wx.EXPAND, 5 )


		sbSizer3.Add( sbSizer31, 1, wx.EXPAND, 5 )


		bSizer2.Add( sbSizer3, 1, wx.EXPAND, 5 )

		bSizer4 = wx.BoxSizer( wx.HORIZONTAL )


		bSizer4.Add( ( 0, 0), 1, wx.EXPAND, 5 )

		self.m_button5 = wx.Button( self, wx.ID_ANY, u"点击一键抠图", wx.Point( -1,-1 ), wx.Size( 120,50 ), 0 )
		self.m_button5.SetFont( wx.Font( 9, wx.FONTFAMILY_MODERN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, "黑体" ) )
		self.m_button5.SetForegroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_CAPTIONTEXT ) )
		self.m_button5.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_INFOBK ) )

		bSizer4.Add( self.m_button5, 0, wx.ALL, 5 )


		bSizer4.Add( ( 0, 0), 1, wx.EXPAND, 5 )


		bSizer2.Add( bSizer4, 1, wx.EXPAND, 5 )


		self.SetSizer( bSizer2 )
		self.Layout()

		self.Centre( wx.BOTH )

	def __del__( self ):
		pass


