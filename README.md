To run the urdf file... 

1 launch an empty world
```python
gz sim empty.sdf
```

2 open a seperate terminal

3 load model

```python
gz service -s /world/empty/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 1000 --req 'sdf_filename: "Stewart.urdf", name: "Stewart_model"'
```
`NOTE: You need to be load the model from the same directory, if you are loading it from a different directory you need to provide the file path... /path/to/model/Stewart.urdf`


Gazebo Harmonic can be installed with ``` sudo apt-get install ros-jazzy-ros-gz ```
