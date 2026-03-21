# -*- coding: UTF-8-*-
# -*- coding: mbcs -*-
#SMPCGLT
#Writen by Wang Haobo .  HIT .Harbin .China
#
from abaqus import *
from abaqusConstants import *
from part import *
from material import *
import section
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
import math
import numpy as np
import random
import itertools
import csv
import os
from odbAccess import *
from odbSection import *

current_directory = os.getcwd()
if os.path.basename(current_directory) != 'SMPCGLT_NonLinearBuckling':
    os.chdir(current_directory)
    folder_name = 'SMPCGLT_NonLinearBuckling'
    if not os.path.exists('SMPCGLT_NonLinearBuckling'):
        os.mkdir('SMPCGLT_NonLinearBuckling')
    os.chdir('SMPCGLT_NonLinearBuckling')

def finite_element_analysis(Sectionalradius1, angle1, radius1 , Lumbus1, Sectionalradius2, angle2, radius2 , Lumbus2 ,jobnum):
    ##Parametric modeling

    print("[Running Analysis] Job {}: ({}, {}, {}, {}) with ({}, {}, {}, {}) starting analysis...".format
          (jobnum, Sectionalradius1, angle1, radius1, Lumbus1, Sectionalradius2, angle2, radius2, Lumbus2))

    Mdb()

    length = 1000
    l1 = length
    composite_thickness = 0.15
    t1 = composite_thickness
    Design_space = 120.0
    materialtable=((120000.0,8800.0, 0.31, 8200.0, 8200.0,7200.0),)
    materialsection=((t1/6, 45), (t1/6, -45), (t1/6, 0), (t1/6, 0), (t1/6, -45), (t1/6, 45))
    meshSize = Design_space/30.0
    cpunum=32
    
    model=mdb.models['Model-1']
    ##Part##
    
    def createside(Sectionalradius,angle, radius,Lumbus,sidename):

        model.ConstrainedSketch(name='__profile__', sheetSize=400.0)

        sr = Sectionalradius
        r1 = radius
        a1 = angle
        w = Lumbus
        r2 = (sr - r1 + r1 * cos(angle * pi / 180.0) )/ (1 - cos(angle * pi / 180.0))
        w1 = Design_space *0.5- angle * r1 * pi / 180 - angle * r2 * pi / 90 - w*0.5

        # Lines
        model.sketches['__profile__'].Line(
            point1=(-w * 0.5, sr),
            point2=(w * 0.5, sr))
        model.sketches['__profile__'].Line(
            point1=(r1 * np.sin(a1 * np.pi / 180.0) + r2 * np.sin(a1 * np.pi / 180.0) + w * 0.5, 0.0),
            point2=(r1 * np.sin(a1 * np.pi / 180.0) + r2 * np.sin(a1 * np.pi / 180.0) + w * 0.5 + w1, 0.0))
        model.sketches['__profile__'].Line(
            point1=(-(r1 * np.sin(a1 * np.pi / 180.0) + r2 * np.sin(a1 * np.pi / 180.0) + w * 0.5), 0.0),
            point2=(-(r1 * np.sin(a1 * np.pi / 180.0) + r2 * np.sin(a1 * np.pi / 180.0) + w * 0.5 + w1), 0.0))
        # Arcs
        ## First quadrant
        model.sketches['__profile__'].ArcByCenterEnds(
            center=(w * 0.5, sr - r1),
            direction=CLOCKWISE,
            point1=(w * 0.5, sr),
            point2=(r1 * np.sin(a1 * np.pi / 180.0) + w * 0.5, sr - r1 + r1 * np.cos(a1 * np.pi / 180.0)))
        model.sketches['__profile__'].ArcByCenterEnds(
            center=(r1 * np.sin(a1 * np.pi / 180.0) + r2 * np.sin(a1 * np.pi / 180.0) + w * 0.5, r2),
            direction=COUNTERCLOCKWISE,
            point1=(r1 * np.sin(a1 * np.pi / 180.0) + w * 0.5, sr - r1 + r1 * np.cos(a1 * np.pi / 180.0)),
            point2=(r1 * np.sin(a1 * np.pi / 180.0) + r2 * np.sin(a1 * np.pi / 180.0) + w * 0.5, 0.0))
        ## Second quadrant
        model.sketches['__profile__'].ArcByCenterEnds(
            center=(-w * 0.5, sr - r1),
            direction=COUNTERCLOCKWISE,
            point1=(-w * 0.5, sr),
            point2=(-(r1 * np.sin(a1 * np.pi / 180.0) + w * 0.5), sr - r1 + r1 * np.cos(a1 * np.pi / 180.0)))
        model.sketches['__profile__'].ArcByCenterEnds(
            center=(-(r1 * np.sin(a1 * np.pi / 180.0) + r2 * np.sin(a1 * np.pi / 180.0) + w * 0.5), r2),
            direction=CLOCKWISE,
            point1=(-(r1 * np.sin(a1 * np.pi / 180.0) + w * 0.5), sr - r1 + r1 * np.cos(a1 * np.pi / 180.0)),
            point2=(-(r1 * np.sin(a1 * np.pi / 180.0) + r2 * np.sin(a1 * np.pi / 180.0) + w * 0.5), 0.0))   
           
        model.Part(dimensionality=THREE_D, name=sidename, type=
            DEFORMABLE_BODY)
        model.parts[sidename].BaseWire(sketch=model.sketches['__profile__'])
        del model.sketches['__profile__']
        
    createside(Sectionalradius1,angle1, radius1 , Lumbus1, 'side1')
    createside(Sectionalradius2,angle2, radius2 , Lumbus2, 'side2')

    model.rootAssembly.Instance(name='side1-1', part=model.parts['side1'], dependent=ON)
    model.rootAssembly.Instance(name='side2-1', part=model.parts['side2'], dependent=ON)
    mdb.models['Model-1'].rootAssembly.translate(instanceList=('side2-1', ), vector=(0.0, 0.0, length))

    model.rootAssembly.InstanceFromBooleanMerge(name='halfcglt', instances=(model.rootAssembly.instances['side1-1'], 
        model.rootAssembly.instances['side2-1'], ), originalInstances=SUPPRESS, domain=GEOMETRY)

    e1 = model.parts['halfcglt'].edges
    model.parts['halfcglt'].ShellLoft(loftsections=((e1[0], e1[1], e1[2], e1[3], e1[4], e1[5], e1[6]), (
        e1[7], e1[8], e1[9], e1[10], e1[11], e1[12], e1[13])), startCondition=NONE, 
        endCondition=NONE)

    del model.rootAssembly.features['side1-1']
    del model.rootAssembly.features['side2-1']

    model.rootAssembly.RadialInstancePattern(instanceList=('halfcglt-1', ), point=(0.0, 0.0, 
        0.0), axis=(0.0, 0.0, 1.0), number=2, totalAngle=180.0)
    
    model.rootAssembly.InstanceFromBooleanMerge(name='cglt', instances=(model.rootAssembly.instances['halfcglt-1'], 
        model.rootAssembly.instances['halfcglt-1-rad-2'], ), originalInstances=SUPPRESS, 
        domain=GEOMETRY)
    
    del model.rootAssembly.features['halfcglt-1']
    del model.rootAssembly.features['halfcglt-1-rad-2']

    ##Material##

    ##set cglt_single and cglt_double
    ###Point coordinates for findat
    ####side1
    radius12 = (Sectionalradius1 - radius1 + radius1 * np.cos(angle1 * np.pi / 180.0) )/ (1 - np.cos(angle1 * np.pi / 180.0))
    w12 = Design_space *0.5- angle1 * radius1 * np.pi / 180 - angle1 * radius12 * np.pi / 90 - Lumbus1*0.5
    #####single
    side1single1x= Lumbus1* 0.5 + radius1 * np.sin(angle1 * np.pi / 360.0) 
    side1single1y= Sectionalradius1 - radius1 + radius1 * np.cos(angle1 * np.pi / 360.0)

    side1single2x= Lumbus1*0.5+ radius1 * np.sin(angle1 * np.pi / 180.0) + radius12 * np.sin(angle1 * np.pi / 180.0) -radius12*np. sin(angle1 * np.pi / 360.0) 
    side1single2y= radius12- radius12 * np.cos(angle1 * np.pi / 360.0)
    #####double
    side1doublex= Lumbus1*0.5+ radius1 * np.sin(angle1 * np.pi / 180.0) + radius12 * np.sin(angle1 * np.pi / 180.0) + w12 * 0.5

    ####side2
    radius22 = (Sectionalradius2 - radius2 + radius2 * np.cos(angle2 * np.pi / 180.0) )/ (1 - np.cos(angle2 * np.pi / 180.0))
    w22 = Design_space *0.5- angle2 * radius2 * np.pi / 180 - angle2 *radius22 * np.pi / 90 - Lumbus2*0.5
    #####single
    side2single1x= Lumbus2* 0.5 + radius2 * np.sin(angle2 * np.pi / 360.0)
    side2single1y= Sectionalradius2 - radius2 + radius2 * np.cos(angle2 * np.pi / 360.0)

    side2single2x= Lumbus2*0.5 + radius2 * np.sin(angle2 * np.pi / 180.0) + radius22 * np.sin(angle2 * np.pi / 180.0) -radius22*np. sin(angle2 * np.pi / 360.0)
    side2single2y= radius22- radius22 * np.cos(angle2 * np.pi / 360.0)
    #####double
    side2doublex= Lumbus2*0.5+radius2 * np.sin(angle2 * np.pi / 180.0) + radius22 * np.sin(angle2 * np.pi / 180.0) + w22 * 0.5
 
    model.parts['cglt'].Set(faces=model.parts['cglt'].faces.findAt(
        ((0.0, Sectionalradius1,0.0),), 
        ((0.0, -Sectionalradius1,0.0),),
        ((side1single1x,side1single1y, 0.0),), 
        ((side1single2x,side1single2y, 0.0),), 
        ((-side1single1x,side1single1y, 0.0),), 
        ((-side1single2x,side1single2y, 0.0),), 
        ((side1single1x,-side1single1y, 0.0),), 
        ((side1single2x,-side1single2y, 0.0),), 
        ((-side1single1x,-side1single1y, 0.0),), 
        ((-side1single2x,-side1single2y, 0.0),),        
        ), name='cglt_single')

    model.parts['cglt'].Set(faces=model.parts['cglt'].faces.findAt(
        ((side1doublex,0.0, 0.0),), 
        ),name='cglt_doub1')
    model.parts['cglt'].Set(faces=model.parts['cglt'].faces.findAt(
        ((-side1doublex,0.0, 0.0),), 
        ),name='cglt_doub2')
    
    ##create material
    model.Material(name='Composite')
    model.materials['Composite'].Elastic(table=materialtable, type=LAMINA)
    model.materials['Composite'].Density(table=((1.6e-09,),))

    ##SectionAssignment
    layup_tuple = tuple(
        section.SectionLayer(
            material='Composite',
            thickness=float(layer[0]),
            orientAngle=float(layer[1])
        ) for layer in materialsection
    )
    model.CompositeShellSection(name='Section-single',idealization=NO_IDEALIZATION,
        integrationRule=SIMPSON,layup=layup_tuple,poissonDefinition=DEFAULT, preIntegrate=OFF,
        symmetric=False, temperature=GRADIENT,thicknessModulus=None, thicknessType=UNIFORM,useDensity=OFF)

    model.parts['cglt'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=model.parts['cglt'].sets['cglt_single'],
        sectionName='Section-single',
        thicknessAssignment=FROM_SECTION)

    symmetric_layup = layup_tuple + tuple(
        section.SectionLayer(
            material='Composite',
            thickness=layer.thickness,
            orientAngle=layer.orientAngle
        ) for layer in reversed(layup_tuple)
    )
    model.CompositeShellSection(
        name='Section-double',
        idealization=NO_IDEALIZATION,
        integrationRule=SIMPSON,
        layup=symmetric_layup,
        poissonDefinition=DEFAULT, preIntegrate=OFF,
        symmetric=True, temperature=GRADIENT,
        thicknessModulus=None, thicknessType=UNIFORM,
        useDensity=OFF)

    model.parts['cglt'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=model.parts['cglt'].sets['cglt_doub1'],
        sectionName='Section-double',
        thicknessAssignment=FROM_SECTION)
    model.parts['cglt'].SectionAssignment(
        offset=0.0,
        offsetField='',
        offsetType=MIDDLE_SURFACE,
        region=model.parts['cglt'].sets['cglt_doub2'],
        sectionName='Section-double',
        thicknessAssignment=FROM_SECTION)

    model.parts['cglt'].MaterialOrientation(additionalRotationType=ROTATION_NONE, axis=AXIS_2, fieldName='',
        localCsys=None, orientationType=GLOBAL, region=model.parts['cglt'].sets['cglt_single'])
    model.parts['cglt'].MaterialOrientation(additionalRotationType=ROTATION_NONE, axis=AXIS_2, fieldName='',
        localCsys=None, orientationType=GLOBAL, region=model.parts['cglt'].sets['cglt_doub1'])
    model.parts['cglt'].MaterialOrientation(additionalRotationType=ROTATION_NONE, axis=AXIS_2, fieldName='',
        localCsys=None, orientationType=GLOBAL, region=model.parts['cglt'].sets['cglt_doub2'])
    
    ##creat referencePoints
    model.rootAssembly.ReferencePoint(point=(0.0, 0.0, 0.0))
    model.rootAssembly.ReferencePoint(point=(0.0, 0.0, l1))

    ##set node
    model.rootAssembly.Set(referencePoints=(model.rootAssembly.referencePoints[11],), name='Reference-Point-1')
    model.rootAssembly.Set(referencePoints=(model.rootAssembly.referencePoints[12],), name='Reference-Point-2')

    ##creat section
    model.rootAssembly.Set(edges=model.rootAssembly.instances['cglt-1'].edges.findAt(
        ((0.0, Sectionalradius1,0.0),), 
        ((0.0, -Sectionalradius1,0.0),),
        ((side1single1x,side1single1y, 0.0),), 
        ((side1single2x,side1single2y, 0.0),), 
        ((-side1single1x,side1single1y, 0.0),), 
        ((-side1single2x,side1single2y, 0.0),), 
        ((side1single1x,-side1single1y, 0.0),), 
        ((side1single2x,-side1single2y, 0.0),), 
        ((-side1single1x,-side1single1y, 0.0),), 
        ((-side1single2x,-side1single2y, 0.0),),
        ((side1doublex,0.0, 0.0),),
        ((-side1doublex,0.0, 0.0),),
        ), name='cglt_section1')
    model.rootAssembly.Set(edges=model.rootAssembly.instances['cglt-1'].edges.findAt(
        ((0.0, Sectionalradius2,l1),), 
        ((0.0, -Sectionalradius2,l1),),
        ((side2single1x,side2single1y, l1),), 
        ((side2single2x,side2single2y, l1),), 
        ((-side2single1x,side2single1y, l1),), 
        ((-side2single2x,side2single2y, l1),), 
        ((side2single1x,-side2single1y, l1),), 
        ((side2single2x,-side2single2y, l1),), 
        ((-side2single1x,-side2single1y, l1),), 
        ((-side2single2x,-side2single2y, l1),),
        ((side2doublex,0.0, l1),),
        ((-side2doublex,0.0, l1),),
        ), name='cglt_section2')
    
    ##Interaction##
    ##create coupling RP_with_cgltsection
    model.Coupling(name='Constraint-1', controlPoint=
        model.rootAssembly.sets['Reference-Point-1'], 
        surface=model.rootAssembly.sets['cglt_section1'], influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
        localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
    model.Coupling(name='Constraint-2', controlPoint=
        model.rootAssembly.sets['Reference-Point-2'], 
        surface=model.rootAssembly.sets['cglt_section2'], influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
        localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
    
    ##mesh##
    model.parts['cglt'].seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=meshSize)
    #model.parts['cglt'].setMeshControls(regions=(model.parts['cglt'].faces,), technique=STRUCTURED)
    model.parts['cglt'].setElementType(elemTypes=(
                ElemType(elemCode=S4R, elemLibrary=EXPLICIT, secondOrderAccuracy=OFF, hourglassControl=STIFFNESS),
               ), regions=(model.parts['cglt'].faces,))
    model.parts['cglt'].generateMesh()

    ##Step##
    model.BuckleStep(name='Step-LinearBuckle', previous='Initial',
        numEigen=20, eigensolver=LANCZOS, minEigen=0.0, blockSize=DEFAULT,maxBlocks=DEFAULT)
    model.fieldOutputRequests['F-Output-1'].setValues(variables=(
        'S', 'U', 'RM'))
    
    ##Load##
    model.DisplacementBC(name='BC-1', createStepName='Initial',region=model.rootAssembly.sets['Reference-Point-1'], 
        u1=SET, u2=SET, u3=SET, ur1=SET,ur2=SET, ur3=SET,
        amplitude=UNSET, distributionType=UNIFORM, fieldName='',localCsys=None)
    model.DisplacementBC(name='BC-2', createStepName='Step-LinearBuckle',region=model.rootAssembly.sets['Reference-Point-2'],   
        u1=UNSET, u2=UNSET, u3=-2.0, ur1=UNSET,ur2=UNSET, ur3=UNSET,
        amplitude=UNSET, distributionType=UNIFORM, fieldName='',localCsys=None)

    ##Add keywords   
    model.keywordBlock.synchVersions()
    lines = list(model.keywordBlock.sieBlocks)  
    target_line = next(i for i, line in enumerate(lines) if "*Restart, write, frequency=0" in line)  
    model.keywordBlock.insert(target_line , "*node file\n u") 
    

    ##Job##
    jobName_Linear = 'cgltLinear_no{}'.format(jobnum)
    mdb.Job(model='Model-1', name=jobName_Linear, numCpus=cpunum, numDomains=cpunum)
    mdb.saveAs(pathName=jobName_Linear)
    mdb.jobs[jobName_Linear].submit()
    mdb.jobs[jobName_Linear].waitForCompletion()

    ##copy model
    mdb.Model(name='Model-NonLinearBuckling', objectToCopy=mdb.models['Model-1'])
    del mdb.models['Model-NonLinearBuckling'].steps['Step-LinearBuckle']

    NLmodel = mdb.models['Model-NonLinearBuckling']

    ##NonLinearStep
    NLmodel.StaticRiksStep(name='Step-NonLinearBuckle',
                                                         previous='Initial', initialArcInc=0.01, minArcInc=1e-08,
                                                         maxArcInc=0.05,
                                                         nlgeom=ON)
    NLmodel.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'U', 'RM'))
    NLmodel.FieldOutputRequest(name='F-Output-2',
                                                             createStepName='Step-NonLinearBuckle',
                                                             variables=('S', 'U', 'UR', 'RM'),
                                                             region=NLmodel.rootAssembly.sets[
                                                                 'Reference-Point-2'],
                                                             sectionPoints=DEFAULT, rebar=EXCLUDE)
    NLmodel.HistoryOutputRequest(name='H-Output-2',
                                                               createStepName='Step-NonLinearBuckle',
                                                               variables=('UR', 'RM'),
                                                               region=NLmodel.rootAssembly.sets[
                                                                   'Reference-Point-2'], sectionPoints=DEFAULT, rebar=EXCLUDE)

    ##NonLinearLoad
    NLmodel.DisplacementBC(name='BC-1', createStepName='Initial',
                                                         region=NLmodel.rootAssembly.sets['Reference-Point-1'], u1=SET, u2=SET, u3=SET, ur1=SET,
                                                         ur2=SET, ur3=SET,
                                                         amplitude=UNSET, distributionType=UNIFORM, fieldName='',
                                                         localCsys=None)
    NLmodel.DisplacementBC(name='BC-2', createStepName='Step-NonLinearBuckle',
                                                         region=NLmodel.rootAssembly.sets['Reference-Point-2'], u1=0.0, u2=UNSET, u3=UNSET,
                                                         ur1=-1.0, ur2=0.0, ur3=0.0,
                                                         amplitude=UNSET, distributionType=UNIFORM, fieldName='',
                                                         localCsys=None)

    ##Add keywords
    NLmodel.keywordBlock.setValues(edited=0)
    NLmodel.keywordBlock.synchVersions(
        storeNodesAndElements=False)
    lines = list(NLmodel.keywordBlock.sieBlocks)
    target_line = next(i for i, line in enumerate(lines) if "** STEP: Step-NonLinearBuckle" in line)
    new_line = "*imperfection,file=cgltLinear_no{},Step=1\n1,0.5e-3\n".format(jobnum)
    NLmodel.keywordBlock.insert(target_line-1, new_line)

    

    ##Job##
    jobName_NonLinear = 'cgltNonLinear_no{}'.format(jobnum)
    mdb.Job(model='Model-NonLinearBuckling', name=jobName_NonLinear, numCpus=cpunum, numDomains=cpunum)
    mdb.saveAs(pathName=jobName_NonLinear)
    mdb.jobs[jobName_NonLinear].submit()
    mdb.jobs[jobName_NonLinear].waitForCompletion()
    mdb.save()




def run_finite_element_jobs(csv_filename='selected_pairs.csv', jobmax=10, start_jobnum=1):
    with open(csv_filename, mode='rb') as file:
        reader = csv.reader(file)
        next(reader) 

        start_line_found = False
        for row in reader:
            jobnum = int(row[8]) 

            if jobnum == start_jobnum:
                start_line_found = True
                print("Starting from jobnum ={}".format(start_jobnum))
                break  

        if not start_line_found:
            print("Error: start_jobnum={} not found in the CSV file.".format(start_jobnum))
            return
     
        while True:
            jobnum = int(row[8])  
            
            if jobnum > jobmax:
                print("Reached the maximum job number, stopping execution.")
                break
  
            Sectionalradius1 = int(row[0])
            angle1 = int(row[1])
            radius1 = int(row[2])
            Lumbus1 = int(row[3])
            Sectionalradius2 = int(row[4])
            angle2 = int(row[5])
            radius2 = int(row[6])
            Lumbus2 = int(row[7])
          
            finite_element_analysis(Sectionalradius1, angle1, radius1, Lumbus1,
                                    Sectionalradius2, angle2, radius2, Lumbus2, jobnum)


if __name__ == "__main__":
    csv_filename = os.path.join('/home', 'selected_pairs.csv')
    jobmax = 5000
    start_jobnum = 1  
    run_finite_element_jobs(csv_filename=csv_filename, jobmax=jobmax, start_jobnum=start_jobnum)



