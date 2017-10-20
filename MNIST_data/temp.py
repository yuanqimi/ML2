import xml.dom.minidom
xmlfile=r"D:\TestTemp\TestMe\ML\1d\beam_sript.xml"
datafile=r"D:\TestTemp\TestMe\ML\1d\parametric_results.txt"
f=open(datafile,"w")

ob_dom=xml.dom.minidom.parse(xmlfile)
xml_root=ob_dom.documentElement
for item in xml_root.getElementsByTagName("configuration"):
    for each in item.getElementsByTagName("parameter"):
        print(each.getAttribute("value"),end="  ")
        f.seek(0,2)
        f.write(str(10*float(each.getAttribute("value")))+",")
    print (item.getElementsByTagName("safetyfactor")[0].getAttribute("minimum"))
    f.seek(0, 2)
    f.write(item.getElementsByTagName("safetyfactor")[0].getAttribute("minimum")+"\n")

f.close()