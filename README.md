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
