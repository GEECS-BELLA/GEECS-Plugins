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
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"HTU Live Postprocessing", pos = wx.DefaultPosition, size = wx.Size( 500,565 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

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


        bSizer1.Add( fgSizer1, 0, wx.EXPAND|wx.LEFT|wx.TOP, 5 )

        self.m_staticline1 = wx.StaticLine( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LI_HORIZONTAL )
        bSizer1.Add( self.m_staticline1, 0, wx.EXPAND |wx.ALL, 5 )

        b_image_analyzers = wx.BoxSizer( wx.VERTICAL )

        self.m_analyze_device_label = wx.StaticText( self, wx.ID_ANY, u"Include these image analyzers", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_analyze_device_label.Wrap( -1 )

        b_image_analyzers.Add( self.m_analyze_device_label, 0, wx.ALL, 5 )

        b_image_analyzers_widgets = wx.BoxSizer( wx.HORIZONTAL )

        m_analyze_device_checklistChoices = [u"1", u"2"]
        self.m_analyze_device_checklist = wx.CheckListBox( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, m_analyze_device_checklistChoices, 0 )
        self.m_analyze_device_checklist.SetMinSize( wx.Size( -1,400 ) )

        b_image_analyzers_widgets.Add( self.m_analyze_device_checklist, 1, wx.ALL, 5 )

        b_image_analyzer_properties = wx.BoxSizer( wx.VERTICAL )

        self.m_image_analyzer_propertyGrid = pg.PropertyGridManager(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.propgrid.PGMAN_DEFAULT_STYLE|wx.propgrid.PG_DESCRIPTION)
        self.m_image_analyzer_propertyGrid.SetExtraStyle( wx.propgrid.PG_EX_MODE_BUTTONS )
        b_image_analyzer_properties.Add( self.m_image_analyzer_propertyGrid, 1, wx.ALL|wx.EXPAND, 5 )

        self.m_staticText5 = wx.StaticText( self, wx.ID_ANY, u"Image analyzer configuration", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText5.Wrap( -1 )

        b_image_analyzer_properties.Add( self.m_staticText5, 0, wx.ALL, 5 )

        self.m_config_filePicker = wx.FilePickerCtrl( self, wx.ID_ANY, wx.EmptyString, u"Select image analyzer configuration file", u"*.*", wx.DefaultPosition, wx.DefaultSize, wx.FLP_USE_TEXTCTRL )
        b_image_analyzer_properties.Add( self.m_config_filePicker, 0, wx.ALL|wx.EXPAND, 5 )

        b_load_save_config = wx.BoxSizer( wx.HORIZONTAL )

        self.m_loadconfig_button = wx.Button( self, wx.ID_ANY, u"Load", wx.DefaultPosition, wx.DefaultSize, 0 )
        b_load_save_config.Add( self.m_loadconfig_button, 0, wx.ALL, 5 )

        self.m_saveconfig_button = wx.Button( self, wx.ID_ANY, u"Save", wx.DefaultPosition, wx.DefaultSize, 0 )
        b_load_save_config.Add( self.m_saveconfig_button, 0, wx.ALL, 5 )


        b_image_analyzer_properties.Add( b_load_save_config, 0, wx.EXPAND, 5 )


        b_image_analyzers_widgets.Add( b_image_analyzer_properties, 2, wx.EXPAND, 5 )


        b_image_analyzers.Add( b_image_analyzers_widgets, 1, wx.EXPAND, 5 )


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
        self.m_loadconfig_button.Bind( wx.EVT_BUTTON, self.m_loadconfig_button_OnButtonClick )
        self.m_saveconfig_button.Bind( wx.EVT_BUTTON, self.m_saveconfig_button_OnButtonClick )

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

    def m_loadconfig_button_OnButtonClick( self, event ):
        event.Skip()

    def m_saveconfig_button_OnButtonClick( self, event ):
        event.Skip()


