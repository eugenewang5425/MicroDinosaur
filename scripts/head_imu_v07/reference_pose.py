"""Reference-pose helper shared by the current modeling scripts."""


def set_reference():
    import bpy

    for obj in bpy.data.objects:
        if "angle_deg" in obj and "reference_deg" in obj and obj.name.startswith(("CTRL_", "DCTL_")):
            obj["angle_deg"] = obj["reference_deg"]
            obj.update_tag()
    bpy.context.view_layer.update()
