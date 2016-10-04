#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

Scene::Scene(string filename) {
    obj_numshapes = 0;
    obj_numpolyverts = NULL;
    obj_polysidx = NULL;
    obj_verts = NULL;
    obj_norms = NULL;
    obj_texts = NULL;
    obj_polyoffsets = NULL;
    obj_polysidxflat = NULL;
    objmesh = NULL;
    polyidxcount = 0;

    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

Scene::~Scene()
{

    if (obj_polysidx != NULL)
    {
        for (int i = 0; i < obj_numshapes; i++)
            delete[] obj_polysidx[i];
        delete[] obj_polysidx;
    }
    if (obj_verts != NULL)
        delete[] obj_verts;
    if (obj_norms != NULL)
        delete[] obj_norms;
    if (obj_texts != NULL)
        delete[] obj_texts;
    if (obj_numpolyverts != NULL)
        delete[] obj_numpolyverts;
    if (objmesh != NULL)
        delete objmesh;
    
    if(obj_polyoffsets != NULL)
        delete[] obj_polyoffsets;
    if(obj_polysidxflat != NULL)
        delete[] obj_polysidxflat;

    polyidxcount = 0;
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
							, 2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}


void Scene::loadObj(string filepath, string mtlpath)
{
    if (objmesh != NULL)
    {
        delete[] obj_polyoffsets;
        delete[] obj_polysidxflat;
        delete[] obj_numpolyverts;
        delete[] obj_polysidx;
        delete[] obj_verts;
        delete[] obj_norms;
        delete[] obj_texts;
        delete objmesh;
    }

    objmesh = new ObjMesh(filepath, mtlpath);

    obj_numshapes = objmesh->shapes.size();
    obj_numpolyverts = new int[obj_numshapes];
    obj_polysidx = new int*[obj_numshapes];
    obj_verts = new float[objmesh->attrib.vertices.size()];
    obj_norms = new float[objmesh->attrib.normals.size()];
    obj_texts = new float[objmesh->attrib.texcoords.size()];



    // vertices
    // get the vertex indices

    for (int i = 0; i < objmesh->attrib.vertices.size(); i++){
        obj_verts[i] = objmesh->attrib.vertices[i];
    }

    for (int i = 0; i < objmesh->attrib.normals.size(); i++) {
        obj_norms[i] = objmesh->attrib.normals[i];
    }

    for (int i = 0; i < objmesh->attrib.texcoords.size(); i++) {
        obj_texts[i] = objmesh->attrib.texcoords[i];
    }


    // polygon idx
    for (int i = 0; i < obj_numshapes; i++)
    {
        obj_numpolyverts[i] = objmesh->shapes[i].mesh.indices.size();
        obj_polysidx[i] = new int[objmesh->shapes[i].mesh.indices.size()];

        // get the polygon indices
        for (int j = 0; j < obj_numpolyverts[i]; j++)
        {
            obj_polysidx[i][j] = objmesh->shapes[i].mesh.indices[j].vertex_index;
        }
    }


    obj_polyoffsets = new int[obj_numshapes];

    polyidxcount = 0;
    for (int i = 0; i < obj_numshapes; i++) {
        polyidxcount += objmesh->shapes[i].mesh.indices.size();
        obj_polyoffsets[i] = objmesh->shapes[i].mesh.indices.size();
    }

    obj_polysidxflat = new int[polyidxcount];


    int iter = 0;
    for (int i = 0; i < obj_numshapes; i++){
        for (int j = 0; j < obj_numpolyverts[i]; j++){
            obj_polysidxflat[iter] = objmesh->shapes[i].mesh.indices[j].vertex_index;
            iter++;
        }
    }

    
    // sanity check print
    int iterator = 0;
    for (int i = 0; i < obj_numshapes; i++)
    {
        printf("loop1\n");
        for (int j = iterator; j < iterator + obj_polyoffsets[i]; j += 3)
        {
            int idx1 = obj_polysidxflat[j];
            int idx2 = obj_polysidxflat[j + 1];
            int idx3 = obj_polysidxflat[j + 2];
            int idxo1 = 3 * idx1;
            int idxo2 = 3 * idx2;
            int idxo3 = 3 * idx3;
            printf("i: %d %d %d : [%.1f %.1f %.1f] [%.1f %.1f %.1f] [%.1f %.1f %.1f]\n",
                idx1,
                idx2,
                idx3,
                obj_verts[idxo1],
                obj_verts[idxo1 + 1],
                obj_verts[idxo1 + 2],
                obj_verts[idxo2],
                obj_verts[idxo2 + 1],
                obj_verts[idxo2 + 2],
                obj_verts[idxo3],
                obj_verts[idxo3 + 1],
                obj_verts[idxo3 + 2]);
        }

        iterator += obj_polyoffsets[i];
    }   
    
}