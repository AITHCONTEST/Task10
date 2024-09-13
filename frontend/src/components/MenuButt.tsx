import img from "../assets/new/menu.svg";
import Button from "./Button";
import { useState } from "react";

const MenuButt = () => {
    const [isActive, setIsActive] = useState(false);

    const dynamicStyle = {
        transform: isActive ? 'rotateZ(90deg)' : 'rotateZ(0deg)',
        transition: 'transform 0.3s ease',
        height: '24px'
        
    };

    const tap = ()=>{
        setIsActive(!isActive);
    }    
    return(
        <div style={dynamicStyle}>
            <Button img={img} tap={tap}  alt={"menu"}/>
        </div>
    );
};

export default MenuButt;