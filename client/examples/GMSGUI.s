// Full example for PACBED thickness workflow providing a GUI in GMS
// based on an active PACBED pattern
// v.1.0, 09.06.2022
// Note: if error 'An image with given name cannot be found' is returned on execution please 
// remove the "1" as an input argument of "ExecutePythonScriptString" function in line 134
// -------------------------------------------

string host = "localhost"
number port = 8230

Class PACBEDGUI: UIFrame 
{ 
  PACBEDGUI(Object self)  Result("\n Created #"  + self.ScriptObjectGetID().hex()) 
  ~PACBEDGUI(Object self) Result("\n Destroyed #"+ self.ScriptObjectGetID().hex()) 
  
  TagGroup DialogTG, choCrystal,choCrystal_Items, cmdGetPACBED, MyRealField, MyIntegerField, txtOrientation_u, txtOrientation_v,txtOrientation_w,lblCrystal, lblOrientation_u, lblOrientation_v, lblOrientation_w, lblHT, lblConva, txtHT, txtConva, lblGetPACBED, lblThickness, lblConfidence, cmdCalcThickness
  image img, outimg
  string py
  
  TagGroup CreatePACBEDGUI(Object self) 
    { 
    DialogTG = DLGCreateDialog("AutoPACBED") 
    choCrystal  = DLGCreateChoice(choCrystal_Items,0) // StringField("TiO2 rutile",20)
    choCrystal.DLGAddChoiceItemEntry("Rutile")
    choCrystal.DLGAddChoiceItemEntry("Strontium Titanate")
    
    txtOrientation_u = DLGCreateIntegerField(0,4)
    txtOrientation_v = DLGCreateIntegerField(0,4)
    txtOrientation_w = DLGCreateIntegerField(1,4)
    cmdGetPACBED   = DLGCreatePushButton("Get PACBED pattern","GetPACBEDButton")
    lblCrystal     = DLGCreateLabel("Crystal:") 
    lblOrientation_u = DLGCreateLabel("u:")
    lblOrientation_v = DLGCreateLabel("v:")
    lblOrientation_w = DLGCreateLabel("w:") 
    lblHT		   = DLGCreateLabel("HT [kV]:").DLGAnchor("West")  
    lblConva	   = DLGCreateLabel("Conv. angle [mrad]:").DLGAnchor("West")  
    lblGetPACBED   = DLGCreateLabel("...",40).DLGIdentifier("#PACBEDName").DLGAnchor("Center")  
    txtHT          = DLGCreateRealField(80,10,2).DLGIdentifier("#HTValueInput") 
    txtConva       = DLGCreateRealField(20.0,10,2) 
    cmdCalcThickness = DLGCreatePushButton("Determine Thickness","CalcThicknessButton").DLGIdentifier("#Calc")
    dlgenabled(cmdCalcThickness,0) 
    lblThickness   = DLGCreateLabel("Thickness: ...",20).DLGIdentifier("#Thickness").DLGAnchor("West") 
    lblConfidence  = DLGCreateLabel("Mistilt: ...",20).DLGIdentifier("#Mistilt").DLGAnchor("West")  
      
	TagGroup grpGetPACBED, grpGetPACBEDItems 
	grpGetPACBED = DLGCreateGroup(grpGetPACBEDItems)  
    grpGetPACBEDItems.DLGAddElement(cmdGetPACBED)   
    grpGetPACBEDItems.DLGAddElement(lblGetPACBED) 
    grpGetPACBED.DLGTableLayOut(1,2,0) 
         
    TagGroup MyGroup, MyGroupItems 
    MyGroup = DLGCreateBox("Crystal and Orientation",MyGroupItems) 
    MyGroupItems.DLGAddElement(lblCrystal) 
    MyGroupItems.DLGAddElement(choCrystal)  
    MyGroupItems.DLGAddElement(lblOrientation_u) 
    MyGroupItems.DLGAddElement(txtOrientation_u)
    MyGroupItems.DLGAddElement(lblOrientation_v) 
    MyGroupItems.DLGAddElement(txtOrientation_v) 
    MyGroupItems.DLGAddElement(lblOrientation_w)  
    MyGroupItems.DLGAddElement(txtOrientation_w)
    MyGroup.DLGTableLayOut(2,5,0) 
     
    TagGroup grpPACBED, grpPACBEDItems      
    grpPACBED = DLGCreateBox("Experimental Settings",grpPACBEDItems) 
    grpPACBEDItems.DLGAddElement(lblHT) 
    grpPACBEDItems.DLGAddElement(txtHT)  
    grpPACBEDItems.DLGAddElement(lblConva) 
    grpPACBEDItems.DLGAddElement(txtConva)  
    grpPACBED.DLGTableLayOut(2,2,0) 
    
    TagGroup grpCalcThickness, grpCalcThicknessItems      
    grpCalcThickness = DLGCreateBox("Calculate",grpCalcThicknessItems) 
    grpCalcThicknessItems.DLGAddElement(cmdCalcThickness)
    grpCalcThicknessItems.DLGAddElement(lblThickness)  
    grpCalcThicknessItems.DLGAddElement(lblConfidence) 
    grpCalcThickness.DLGTableLayOut(1,3,0) 
            
    DialogTG.DLGAddElement(grpGetPACBED)
    DialogTG.DLGAddElement(MyGroup) 
    DialogTG.DLGAddElement(grpPACBED) 
    DialogTG.DLGAddElement(grpCalcThickness) 
     
    return DialogTG 
    }   
 
  Object Init(Object self)  return self.super.Init(self.CreatePACBEDGUI()) 
  
  TagGroup GetPACBEDButton(Object self) {
  
	img := GetFrontImage()
	
	If (!img.ImageIsValid()) Exit(0)
	self.LookUpElement("#PACBEDName").DLGTitle(""+img.GetName())
	self.SetElementIsEnabled("#Calc",1) 

	Number sx, sy
	getsize(img,sx,sy)
	Result("\n"+ sx + "/" + sy + " " + img.GetName())
	
	taggroup imgtags=img.imagegettaggroup()
	number HTvalue
	string targettaggroup="Microscope Info:Voltage"
	if(!TagGroupDoesTagExist(imgtags,targettaggroup)){
		showalert("The taggroup : '"+targettaggroup+"' was not found.",2)
		exit(0)
	}
	imgtags.taggroupgettagasnumber(targettaggroup,HTvalue)
	HTvalue = HTvalue / 1000.0
	self.LookUpElement("#HTValueInput").DLGValue(HTvalue) 
	Result("\n"+ HTvalue)
  }
  
  TagGroup CalcThicknessButton(Object self) {
	
	number imgid = img.ImageGetID()
	number ht_entered = txtHT.DLGGetStringValue().val()*1000
	
	string material
	choCrystal.DLGGetNthLabel( choCrystal.DLGGetValue(), material )
	
	number convergence_angle = txtConva.DLGGetStringValue().val()
	
	number zone_u_value = txtOrientation_u.DLGGetStringValue().val()
	number zone_v_value = txtOrientation_v.DLGGetStringValue().val()
	number zone_w_value = txtOrientation_w.DLGGetStringValue().val()
	
  	string pyScript = "from pacbedclient import imagefromresponse, query, arrayfromID; "
  	pyScript += "imagefromresponse(DM, query(image_array=arrayfromID(DM, " + imgid + "), "
  	pyScript += "crystal_structure='" + material + "', acceleration_voltage=" + ht_entered + ", "
  	pyScript += "convergence_angle=" + convergence_angle + ", "
	pyScript += "zone_u=" + zone_u_value + ", zone_v=" + zone_v_value + ", zone_w=" + zone_w_value + ", "
  	pyScript += "host='" + host + "', port=" + port + "))"

	ExecutePythonScriptString( pyScript, 1) // remove 1 if error 'An image with given name cannot be found'

	image r_img := GetNamedImage("pacbed:viz_r")
	image g_img := GetNamedImage("pacbed:viz_g")
	image b_img := GetNamedImage("pacbed:viz_b")

	image validation := rgb(r_img, g_img, b_img)
	validation.ShowImage()

	TagGroup sourcetags = imagegettaggroup(r_img)
	TagGroup targettags = imagegettaggroup(validation)

	taggroupcopytagsfrom(targettags,sourcetags)
	
	number tagvalue_thickness
	string targettaggroup_thickness = "thickness"
	sourcetags.taggroupgettagasnumber(targettaggroup_thickness,tagvalue_thickness)
	self.LookUpElement("#Thickness").DLGTitle("Thickness: " +  Format( tagvalue_thickness/10, "%.1f" ) + " nm")

	number tagvalue_mistilt
	string targettaggroup_mistilt = "mistilt"
	sourcetags.taggroupgettagasnumber(targettaggroup_mistilt,tagvalue_mistilt)
	self.LookUpElement("#Mistilt").DLGTitle("Mistilt: " + Format( tagvalue_mistilt, "%.1f" ) + " mrad")


	DeleteImage(r_img)
	DeleteImage(g_img)
	DeleteImage(b_img)
  }
}
Object DialogOBJ = Alloc(PACBEDGUI).Init() 
	
DialogOBJ.Display("Remote PACBED GUI") 
