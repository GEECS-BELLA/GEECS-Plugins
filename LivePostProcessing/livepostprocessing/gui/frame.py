# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 3.10.1-0-g8feb16b3)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc
import wx.propgrid as pg

###########################################################################
## Class MainFrame
###########################################################################

class MainFrame ( wx.Frame ):

    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Live Run Analysis", pos = wx.DefaultPosition, size = wx.Size( 500,565 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

        self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

        bSizer1 = wx.BoxSizer( wx.VERTICAL )

        fgSizer1 = wx.FlexGridSizer( 0, 4, 0, 0 )
        fgSizer1.SetFlexibleDirection( wx.BOTH )
        fgSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

        self.m_runID_label = wx.StaticText( self, wx.ID_ANY, u"Run ID (yy_mmdd)", wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
        self.m_runID_label.Wrap( -1 )

        fgSizer1.Add( self.m_runID_label, 0, wx.ALL, 5 )

        self.m_runID_text = wx.TextCtrl( self, wx.ID_ANY, u"23_0101", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_runID_text.SetMaxSize( wx.Size( 65,-1 ) )

        fgSizer1.Add( self.m_runID_text, 0, wx.ALIGN_LEFT|wx.ALL, 5 )

        self.m_run_live_analysis_Button = wx.Button( self, wx.ID_ANY, u"Start live analysis", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_run_live_analysis_Button.SetToolTip( u"Watch the run folder for new scans and analyze them when found" )

        fgSizer1.Add( self.m_run_live_analysis_Button, 1, wx.ALL|wx.EXPAND, 5 )

        self.m_stop_live_analysis_Button = wx.Button( self, wx.ID_ANY, u"Stop", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_stop_live_analysis_Button.Enable( False )

        fgSizer1.Add( self.m_stop_live_analysis_Button, 0, wx.ALL, 5 )

        self.m_scanNumber_staticText = wx.StaticText( self, wx.ID_ANY, u"Scan number", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_scanNumber_staticText.Wrap( -1 )

        fgSizer1.Add( self.m_scanNumber_staticText, 0, wx.ALL, 5 )

        self.m_scanNumber_textCtrl = wx.TextCtrl( self, wx.ID_ANY, u"0", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_scanNumber_textCtrl.SetMaxSize( wx.Size( 65,-1 ) )

        fgSizer1.Add( self.m_scanNumber_textCtrl, 0, wx.ALL, 5 )

        self.m_run_scan_analysis_Button = wx.Button( self, wx.ID_ANY, u"Run scan analysis", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_run_scan_analysis_Button.SetToolTip( u"Run offline analysis on a single scan" )

        fgSizer1.Add( self.m_run_scan_analysis_Button, 0, wx.ALL|wx.EXPAND, 5 )


        bSizer1.Add( fgSizer1, 1, wx.EXPAND|wx.LEFT|wx.TOP, 5 )

        self.m_staticline1 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        bSizer1.Add( self.m_staticline1, 0, wx.EXPAND |wx.ALL, 5 )

        b_image_analyzers = wx.BoxSizer( wx.VERTICAL )

        self.m_analyze_device_label = wx.StaticText( self, wx.ID_ANY, u"Include these image analyzers", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_analyze_device_label.Wrap( -1 )

        b_image_analyzers.Add( self.m_analyze_device_label, 0, wx.ALL, 5 )

        bSizer6 = wx.BoxSizer( wx.HORIZONTAL )

        m_analyze_device_checklistChoices = [u"1", u"2"]
        self.m_analyze_device_checklist = wx.CheckListBox( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, m_analyze_device_checklistChoices, 0 )
        self.m_analyze_device_checklist.SetMinSize( wx.Size( -1,400 ) )

        bSizer6.Add( self.m_analyze_device_checklist, 2, wx.ALL, 5 )

        self.m_image_analyzer_propertyGrid = pg.PropertyGrid(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.propgrid.PG_DEFAULT_STYLE)
        bSizer6.Add( self.m_image_analyzer_propertyGrid, 2, wx.ALL, 5 )


        b_image_analyzers.Add( bSizer6, 1, wx.EXPAND, 5 )

        bBackground = wx.BoxSizer( wx.HORIZONTAL )

        self.m_background_label = wx.StaticText( self, wx.ID_ANY, u"Background", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_background_label.Wrap( -1 )

        self.m_background_label.Hide()

        bBackground.Add( self.m_background_label, 0, wx.ALL, 5 )

        self.m_background_filePicker = wx.FilePickerCtrl( self, wx.ID_ANY, wx.EmptyString, u"Select a file", u"*.*", wx.DefaultPosition, wx.DefaultSize, wx.FLP_DEFAULT_STYLE )
        self.m_background_filePicker.Hide()

        bBackground.Add( self.m_background_filePicker, 1, wx.ALL, 5 )


        b_image_analyzers.Add( bBackground, 0, wx.EXPAND, 5 )


        bSizer1.Add( b_image_analyzers, 1, wx.BOTTOM|wx.EXPAND|wx.LEFT, 5 )


        self.SetSizer( bSizer1 )
        self.Layout()
        self.m_statusBar1 = self.CreateStatusBar( 1, wx.STB_SIZEGRIP, wx.ID_ANY )

        self.Centre( wx.BOTH )

        # Connect Events
        self.m_run_live_analysis_Button.Bind( wx.EVT_BUTTON, self.m_run_live_analysis_Button_OnButtonClick )
        self.m_stop_live_analysis_Button.Bind( wx.EVT_BUTTON, self.m_stop_live_analysis_Button_OnButtonClick )
        self.m_run_scan_analysis_Button.Bind( wx.EVT_BUTTON, self.m_run_scan_analysis_Button_OnButtonClick )
        self.m_analyze_device_checklist.Bind( wx.EVT_LISTBOX, self.m_analyze_device_checklist_OnCheckListBoxSelect )
        self.m_analyze_device_checklist.Bind( wx.EVT_CHECKLISTBOX, self.m_analyze_device_checklist_OnCheckListBoxToggled )
        self.m_image_analyzer_propertyGrid.Bind( pg.EVT_PG_CHANGED, self.m_image_analyzer_propertyGrid_OnPropertyGridChanged )
        self.m_background_filePicker.Bind( wx.EVT_FILEPICKER_CHANGED, self.m_background_filePicker_OnFileChanged )

    def __del__( self ):
        pass


    # Virtual event handlers, override them in your derived class
    def m_run_live_analysis_Button_OnButtonClick( self, event ):
        event.Skip()

    def m_stop_live_analysis_Button_OnButtonClick( self, event ):
        event.Skip()

    def m_run_scan_analysis_Button_OnButtonClick( self, event ):
        event.Skip()

    def m_analyze_device_checklist_OnCheckListBoxSelect( self, event ):
        event.Skip()

    def m_analyze_device_checklist_OnCheckListBoxToggled( self, event ):
        event.Skip()

    def m_image_analyzer_propertyGrid_OnPropertyGridChanged( self, event ):
        event.Skip()

    def m_background_filePicker_OnFileChanged( self, event ):
        event.Skip()

