Here is the structured skill file derived from the provided ontology and taxonomy. This format is designed for ingestion by an autonomous AI agent, perception pipeline, or simulation framework.

---

# **Skill Definition: Autonomous Driving Semantic Ontology & Taxonomy**

## **Domain Overview**

* This skill provides a machine-readable vocabulary to describe the driving environment for Automated Driving Systems (ADS) and Advanced Driver Assistance Systems (ADAS).


* It transitions validation from mileage accumulation to scenario-based testing.



## **1. Supported Standards & Frameworks**

* **ISO 34501**: Defines the core vocabulary for test scenarios, distinguishing between static scenes and temporal scenarios.


* **ISO 34502**: Establishes a scenario-based safety evaluation framework based on scenario criticality.


* **ISO 34503**: Specifies the Operational Design Domain (ODD), detailing environmental, geometric, and geographic conditions.


* **ISO 21448 (SOTIF)**: Identifies rare triggering conditions and functional insufficiencies for the Safety of the Intended Functionality.


* **ASAM OpenX Suite**: Utilizes OpenDRIVE for static road networks, OpenSCENARIO for dynamic behaviors, OpenLABEL for multi-sensor data annotation, and OpenXOntology for semantic relationships.



## **2. Entity Classification Taxonomy**

### **A. Road Users & Kinematic Profiles**

* **Passenger Vehicles**: Passenger cars act as the base model for gap acceptance. SUVs create occlusion zones for forward-facing cameras. Vans have minimal rear visibility. Pickup trucks have variable mass based on payloads. Taxis execute highly unpredictable route behaviors.


* **Heavy & Commercial Vehicles**: Buses block entire lanes during stops. School buses trigger strict longitudinal stopping rules. Heavy trucks require avoidance in lateral blind spots. Semi-trailers mandate extended lateral safety buffers due to trailer sweep.


* **Emergency Vehicles**: Police vehicles, fire trucks, and ambulances possess extreme acceleration or mass and mandate immediate yielding or multi-agent negotiation.


* **Vulnerable Road Users (VRUs)**: Pedestrians require continuous pose estimation. Children pose a high risk of occlusion behind parked cars. Wheelchair users require robust LiDAR detection due to atypical radar signatures. Skaters have highly unpredictable trajectory envelopes.


* **Micro-Mobility**: Cyclists require recognition of hand signals for turning. E-bikes feature the visual signature of a bicycle but the acceleration of a scooter.


* **Animals**: Domestic animals execute rapid startle responses. Wildlife herd behavior implies that detecting one animal strongly predicts others. Livestock can completely obstruct roadways.



### **B. Static Road Objects (The Scenery)**

* **Drivable Surfaces**: Includes roads, lanes, shoulders (for minimal risk maneuvers), bike lanes, bus lanes, intersections, roundabouts, highway ramps, and parking spaces.


* **Physical Constraints & Barriers**: Curbs act as hard boundaries for path planners. Sidewalks are the primary domain of VRUs. Guardrails are high-consequence collision objects. Concrete barriers create multipath radar interference.


* **Vertical Infrastructure**: Utility poles serve as reliable localization landmarks. Street lights can cause localized glare or sensor oversaturation.


* **Pavement Markings**: Degradation of lane markings (e.g., from snow) leads to functional insufficiencies. Stop lines trigger mandatory longitudinal deceleration algorithms.



### **C. Dynamic & Transitory Objects**

* **Uncontrolled Hazards**: Falling objects or debris necessitate rapid free-space recalculation.


* **Temporary Infrastructure**: Construction equipment overrides standard HD maps and forces reliance on real-time sensor fusion.


* **Automated Agents**: Delivery robots lack standard human communication cues. Drones confound standard 2D bird's-eye-view perception projections.



## **3. Activities and Semantic Maneuvers**

* **Vehicle Activities**: Rapid braking from a lead vehicle triggers emergency deceleration protocols. Open doors instantly redefine drivable free space.


* **Pedestrian Activities**: Walking speed determines the predicted intersection point with the ego vehicle's path. Jaywalking triggers immediate emergency deceleration.


* **Traffic Control Elements**: Regulatory signs (e.g., Stop, Yield) enforce mandatory constraints. Variable message signs (VMS) require complex natural language processing (NLP) to interpret. Flaggers and police gestures supercede all static lights and signs.



## **4. Operational Design Domain (ODD) Context**

* **Weather & Visibility**: Rain and snow distort camera lenses and obscure lane markings. Fog and smoke scatter LiDAR pulses, forcing heavier reliance on radar.


* **Surface & Force**: Ice reduces tire-road friction, expanding required stopping distances. High crosswinds require compensatory steering torque.


* **Lighting**: Dawn and dusk create severe horizontal solar glare, which is a classic SOTIF triggering condition.



## **5. Semantic Ontology Graph**

* The system utilizes ontological mapping to reason about physical and legal dependencies.


* **Vehicle -> is_cutting_into -> Lane**: Triggers deceleration to restore safety margins.


* **Pedestrian -> intends_to_cross -> Crosswalk**: Mandates yielding the right-of-way.


* **Weather (Ice) -> degrades -> Road (Surface)**: Updates global friction coefficient variables.


* **Police Officer -> overrides -> Traffic Light**: Forces the system to ignore the traffic light state.



---

## **6. Machine-Readable Integration Schema (ASAM OpenLABEL)**

The following JSON schema can be directly ingested into your project's perception pipeline to model SOTIF corner cases, such as a pedestrian crossing in heavy rain.

```json
{
  "openlabel": {
    "metadata": {
      "schema_version": "1.0.0",
      "scenario_name": "Urban_Pedestrian_Crossing_Rain_SOTIF",
      "description": "Ego vehicle encounters a jaywalking pedestrian in heavy rain, triggering an RSS braking maneuver.",
      "ontologies": {
        "asam_openx": "https://www.asam.net/standards/asam-openxontology"
      }
    },
    "contexts": {
      "env_001": {
        "name": "Heavy_Rain",
        "type": "EnvironmentalCondition",
        "attributes": {
          "precipitation_rate": "15.0 mm/hr",
          "visibility_reduction": true,
          "surface_friction_coefficient": 0.5
        }
      },
      "odd_001": {
        "name": "Urban_Arterial",
        "type": "OperationalDesignDomain",
        "attributes": {
          "speed_limit": "50 km/h",
          "road_type": "CityStreet",
          "region": "US_RightHandDrive"
        }
      }
    },
    "objects": {
      "ego_vehicle": {
        "name": "Ego",
        "type": "Vehicle",
        "sub_type": "PassengerCar",
        "kinematics": {
          "velocity_x": 12.5,
          "acceleration_x": -4.0
        }
      },
      "ped_001": {
        "name": "Pedestrian_Adult",
        "type": "VulnerableRoadUser",
        "sub_type": "Pedestrian",
        "attributes": {
          "pose": "Walking",
          "is_occluded": false
        }
      },
      "infra_001": {
        "name": "Main_Crosswalk",
        "type": "StaticRoadObject",
        "sub_type": "Crosswalk"
      }
    },
    "actions": {
      "act_001": {
        "type": "Emergency_Braking",
        "actor": "ego_vehicle",
        "trigger": "rss_longitudinal_distance_violation"
      },
      "act_002": {
        "type": "Jaywalking",
        "actor": "ped_001",
        "description": "Pedestrian crossing outside designated crosswalk infra_001."
      }
    },
    "relations": {
      "rel_001": {
        "type": "interacts_with",
        "subject": "ego_vehicle",
        "object": "ped_001"
      },
      "rel_002": {
        "type": "modifies",
        "subject": "env_001",
        "object": "ego_vehicle",
        "description": "Rain reduces optimal braking friction, increasing required RSS stopping distance."
      }
    },
    "frames": {
      "0": {
        "timestamp": 1690384512.001,
        "frame_properties": {
          "sotif_triggering_condition": "High_Rain_Glare"
        },
        "object_data": {
          "ped_001": {
            "bbox_3d": [15.2, 3.1, 0.0, 0.6, 0.6, 1.7, 1.57]
          }
        }
      }
    }
  }
}

```